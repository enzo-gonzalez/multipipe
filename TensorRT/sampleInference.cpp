/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "NvInfer.h"

#include "logger.h"
#include "buffers.h"
#include "sampleDevice.h"
#include "sampleInference.h"
#include "sampleOptions.h"
#include "sampleReporting.h"
#include "sampleUtils.h"
#include<QtDebug>
namespace sample
{

void InitInfer(InferenceEnvironment& iEnv, const InferenceOptions& inference)
{



    for (int s = 0; s < inference.streams; ++s)
    {
        iEnv.context.emplace_back(iEnv.engine->createExecutionContext());
        iEnv.bindings.emplace_back(new Bindings);
    }


    if (iEnv.profiler)
    {
        iEnv.context.front()->setProfiler(iEnv.profiler.get());
    }
}

bool setUpInference(InferenceEnvironment& iEnv, const InferenceOptions& inference)
{
    static bool first=true;

    if(first){
        for (int s = 0; s < inference.streams; ++s)
        {
            iEnv.context.emplace_back(iEnv.engine->createExecutionContext());
            iEnv.bindings.emplace_back(new Bindings);
        }

        if (iEnv.profiler)
        {
            iEnv.context.front()->setProfiler(iEnv.profiler.get());
        }
        iEnv.bindings[0]->Allocation(0, "input", true,
                 (224*224*3*18)/*batch x channel x image (h x w)*/,
                 nvinfer1::DataType::kFLOAT,

                 "fileName");
        qDebug()<<"for first time.............";
        first=false;
    }


//    for (int s = 0; s < inference.streams; ++s)
//    {
//        iEnv.bindings.emplace_back(new Bindings);
//    }

//    if (iEnv.profiler)
//    {
//        iEnv.context.front()->setProfiler(iEnv.profiler.get());
//    }

    const int nOptProfiles = iEnv.engine->getNbOptimizationProfiles();
    const int nBindings = iEnv.engine->getNbBindings();
    const int bindingsInProfile = nOptProfiles > 0 ? nBindings / nOptProfiles : 0;
    const int endBindingIndex = bindingsInProfile ? bindingsInProfile : iEnv.engine->getNbBindings();

    if (nOptProfiles > 1)
    {
        sample::gLogWarning << "Multiple profiles are currently not supported. Running with one profile." << std::endl;
    }

    // Set all input dimensions before all bindings can be allocated
    for (int b = 0; b < endBindingIndex; ++b)
    {
        if (iEnv.engine->bindingIsInput(b))
        {
            auto shape = inference.shapes.find(iEnv.engine->getBindingName(b));

            // If no shape is provided, set dynamic dimensions to 1.
            std::vector<int> staticDims;

            staticDims = shape->second;

            if (!iEnv.context[0]->setBindingDimensions(b, toDims(staticDims)))
            {
                return false;
            }


        }
    }



    return true;
}

namespace
{

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

//!
//! \struct SyncStruct
//! \brief Threads synchronization structure
//!
struct SyncStruct
{
    std::mutex mutex;
    TrtCudaStream mainStream;
    TrtCudaEvent gpuStart{cudaEventBlockingSync};
    TimePoint cpuStart{};
    int sleep{0};
};

struct Enqueue
{
    explicit Enqueue(nvinfer1::IExecutionContext& context, void** buffers)
        : mContext(context)
        , mBuffers(buffers)
    {
    }

    nvinfer1::IExecutionContext& mContext;
    void** mBuffers{};
};

//!
//! \class EnqueueImplicit
//! \brief Functor to enqueue inference with implict batch
//!
class EnqueueImplicit : private Enqueue
{

public:
    explicit EnqueueImplicit(nvinfer1::IExecutionContext& context, void** buffers, int batch)
        : Enqueue(context, buffers)
        , mBatch(batch)
    {
    }

    bool operator()(TrtCudaStream& stream) const
    {
        return mContext.enqueue(mBatch, mBuffers, stream.get(), nullptr);
    }

private:
    int mBatch;
};

//!
//! \class EnqueueExplicit
//! \brief Functor to enqueue inference with explict batch
//!
class EnqueueExplicit : private Enqueue
{

public:
    explicit EnqueueExplicit(nvinfer1::IExecutionContext& context, void** buffers)
        : Enqueue(context, buffers)
    {
    }

    bool operator()(TrtCudaStream& stream) const
    {
        return mContext.enqueueV2(mBuffers, stream.get(), nullptr);
    }
};

//!
//! \class EnqueueGraph
//! \brief Functor to enqueue inference from CUDA Graph
//!
class EnqueueGraph
{

public:
    explicit EnqueueGraph(TrtCudaGraph& graph)
        : mGraph(graph)
    {
    }

    bool operator()(TrtCudaStream& stream) const
    {
        return mGraph.launch(stream);
    }

    TrtCudaGraph& mGraph;
};

using EnqueueFunction = std::function<bool(TrtCudaStream&)>;

enum class StreamType : int
{
    kINPUT = 0,
    kCOMPUTE = 1,
    kOUTPUT = 2,
    kNUM = 3
};

enum class EventType : int
{
    kINPUT_S = 0,
    kINPUT_E = 1,
    kCOMPUTE_S = 2,
    kCOMPUTE_E = 3,
    kOUTPUT_S = 4,
    kOUTPUT_E = 5,
    kNUM = 6
};

using MultiStream = std::array<TrtCudaStream, static_cast<int>(StreamType::kNUM)>;

using MultiEvent = std::array<std::unique_ptr<TrtCudaEvent>, static_cast<int>(EventType::kNUM)>;

using EnqueueTimes = std::array<TimePoint, 2>;

//!
//! \class Iteration
//! \brief Inference iteration and streams management
//!
class Iteration
{

public:
    Iteration(int id, const InferenceOptions& inference, nvinfer1::IExecutionContext& context, Bindings& bindings)
        : mBindings(bindings)
        , mStreamId(id)
        , mDepth(1 + inference.overlap)
        , mActive(mDepth)
        , mEvents(mDepth)
        , mEnqueueTimes(mDepth)
    {
        for (int d = 0; d < mDepth; ++d)
        {
            for (int e = 0; e < static_cast<int>(EventType::kNUM); ++e)
            {
                mEvents[d][e].reset(new TrtCudaEvent(!inference.spin));
            }
        }
        createEnqueueFunction(inference, context, bindings);
    }

    void WaitMs(int ms)
    {
        QEventLoop q;
        QTimer tT;
        tT.setSingleShot(true);
        QObject::connect(&tT, SIGNAL(timeout()), &q, SLOT(quit()));
        tT.start(ms);
        q.exec();
        if(tT.isActive()){
            tT.stop();
        } else {

        }
    }

    bool query(bool skipTransfers)
    {
        if (mActive[mNext])
        {
            return true;
        }

        if (!skipTransfers)
        {
            record(EventType::kINPUT_S, StreamType::kINPUT);
            mBindings.transferInputToDevice(getStream(StreamType::kINPUT));
            record(EventType::kINPUT_E, StreamType::kINPUT);
            wait(EventType::kINPUT_E, StreamType::kCOMPUTE); // Wait for input DMA before compute
        }

        record(EventType::kCOMPUTE_S, StreamType::kCOMPUTE);
        recordEnqueueTime();
        if (!mEnqueue(getStream(StreamType::kCOMPUTE)))
        {
            return false;
        }
        recordEnqueueTime();
        record(EventType::kCOMPUTE_E, StreamType::kCOMPUTE);

        if (!skipTransfers)
        {
            wait(EventType::kCOMPUTE_E, StreamType::kOUTPUT); // Wait for compute before output DMA
            record(EventType::kOUTPUT_S, StreamType::kOUTPUT);
            mBindings.transferOutputToHost(getStream(StreamType::kOUTPUT));
            record(EventType::kOUTPUT_E, StreamType::kOUTPUT);


        }

        mActive[mNext] = true;
        moveNext();
        return true;
    }

    float sync(
            const TimePoint& cpuStart, const TrtCudaEvent& gpuStart, std::vector<InferenceTrace>& trace, bool skipTransfers)
    {
        if (mActive[mNext])
        {
            if (skipTransfers)
            {
                getEvent(EventType::kCOMPUTE_E).synchronize();
            }
            else
            {
                getEvent(EventType::kOUTPUT_E).synchronize();
            }
            trace.emplace_back(getTrace(cpuStart, gpuStart, skipTransfers));
            mActive[mNext] = false;
            return getEvent(EventType::kCOMPUTE_S) - gpuStart;
        }
        return 0;
    }

    void syncAll(
            const TimePoint& cpuStart, const TrtCudaEvent& gpuStart, std::vector<InferenceTrace>& trace, bool skipTransfers)
    {
        for (int d = 0; d < mDepth; ++d)
        {
            sync(cpuStart, gpuStart, trace, skipTransfers);
            moveNext();
        }
    }

    void wait(TrtCudaEvent& gpuStart)
    {
        getStream(StreamType::kINPUT).wait(gpuStart);
    }

    void setInputData()
    {
        mBindings.transferInputToDevice(getStream(StreamType::kINPUT));
    }

    void fetchOutputData()
    {
        mBindings.transferOutputToHost(getStream(StreamType::kOUTPUT));
    }

private:
    void moveNext()
    {
        mNext = mDepth - 1 - mNext;
    }

    TrtCudaStream& getStream(StreamType t)
    {
        return mStream[static_cast<int>(t)];
    }

    TrtCudaEvent& getEvent(EventType t)
    {
        return *mEvents[mNext][static_cast<int>(t)];
    }

    void record(EventType e, StreamType s)
    {
        getEvent(e).record(getStream(s));
    }

    void recordEnqueueTime()
    {
        mEnqueueTimes[mNext][enqueueStart] = std::chrono::high_resolution_clock::now();
        enqueueStart = 1 - enqueueStart;
    }

    TimePoint getEnqueueTime(bool start)
    {
        return mEnqueueTimes[mNext][start ? 0 : 1];
    }

    void wait(EventType e, StreamType s)
    {
        getStream(s).wait(getEvent(e));
    }

    InferenceTrace getTrace(const TimePoint& cpuStart, const TrtCudaEvent& gpuStart, bool skipTransfers)
    {
        float is
                = skipTransfers ? getEvent(EventType::kCOMPUTE_S) - gpuStart : getEvent(EventType::kINPUT_S) - gpuStart;
        float ie
                = skipTransfers ? getEvent(EventType::kCOMPUTE_S) - gpuStart : getEvent(EventType::kINPUT_E) - gpuStart;
        float os
                = skipTransfers ? getEvent(EventType::kCOMPUTE_E) - gpuStart : getEvent(EventType::kOUTPUT_S) - gpuStart;
        float oe
                = skipTransfers ? getEvent(EventType::kCOMPUTE_E) - gpuStart : getEvent(EventType::kOUTPUT_E) - gpuStart;

        return InferenceTrace(mStreamId,
                              std::chrono::duration<float, std::milli>(getEnqueueTime(true) - cpuStart).count(),
                              std::chrono::duration<float, std::milli>(getEnqueueTime(false) - cpuStart).count(), is, ie,
                              getEvent(EventType::kCOMPUTE_S) - gpuStart, getEvent(EventType::kCOMPUTE_E) - gpuStart, os, oe);
    }

    void createEnqueueFunction(
            const InferenceOptions& inference, nvinfer1::IExecutionContext& context, Bindings& bindings)
    {
        if (inference.batch)
        {
            mEnqueue = EnqueueFunction(EnqueueImplicit(context, mBindings.getDeviceBuffers(), inference.batch));
        }
        else
        {
            mEnqueue = EnqueueFunction(EnqueueExplicit(context, mBindings.getDeviceBuffers()));
        }
        if (inference.graph)
        {
            TrtCudaStream& stream = getStream(StreamType::kCOMPUTE);
            // Avoid capturing initialization calls by executing the enqueue function at least
            // once before starting CUDA graph capture.
            const auto ret = mEnqueue(stream);
            assert(ret);
            stream.synchronize();

            mGraph.beginCapture(stream);
            // The built TRT engine may contain operations that are not permitted under CUDA graph capture mode.
            // When the stream is capturing, the enqueue call may return false if the current CUDA graph capture fails.
            if (mEnqueue(stream))
            {
                mGraph.endCapture(stream);
                mEnqueue = EnqueueFunction(EnqueueGraph(mGraph));
            }
            else
            {
                mGraph.endCaptureOnError(stream);
                // Ensure any CUDA error has been cleaned up.
                cudaCheck(cudaGetLastError());
                sample::gLogWarning << "The built TensorRT engine contains operations that are not permitted under "
                                       "CUDA graph capture mode."
                                    << std::endl;
                sample::gLogWarning << "The specified --useCudaGraph flag has been ignored. The inference will be "
                                       "launched without using CUDA graph launch."
                                    << std::endl;
            }
        }
    }

    Bindings& mBindings;

    TrtCudaGraph mGraph;
    EnqueueFunction mEnqueue;

    int mStreamId{0};
    int mNext{0};
    int mDepth{2}; // default to double buffer to hide DMA transfers

    std::vector<bool> mActive;
    MultiStream mStream;
    std::vector<MultiEvent> mEvents;

    int enqueueStart{0};
    std::vector<EnqueueTimes> mEnqueueTimes;
};

using IterationStreams = std::vector<std::unique_ptr<Iteration>>;
void WaitMs(int ms)
{
    QEventLoop q;
    QTimer tT;
    tT.setSingleShot(true);
    QObject::connect(&tT, SIGNAL(timeout()), &q, SLOT(quit()));
    tT.start(ms);
    q.exec();
    if(tT.isActive()){
        tT.stop();
    } else {

    }
}
bool inferenceLoop(IterationStreams& iStreams, const TimePoint& cpuStart, const TrtCudaEvent& gpuStart, int iterations,
                   float maxDurationMs, float warmupMs, std::vector<InferenceTrace>& trace, bool skipTransfers)
{
    maxDurationMs = 1;
    skipTransfers=false;

    //    for (int i = 0; i < 2; i++)
    //    {
    if(!iStreams[0]->query(skipTransfers))

    {
        std::cout<<"error querry skip transfer:"<<skipTransfers<<std::endl;
        return false;
    }
    //    }
        iStreams[0]->syncAll(cpuStart, gpuStart, trace, skipTransfers);


    return true;
}
bool verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = 10;
    float* output = static_cast<float*>(buffers.getHostBuffer("Plus214_Output_0"));
    float val{0.0f};
    int idx{0};

    // Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < outputSize; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    //   sample::gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i])
        {
            idx = i;
        }

    }

    return 0;//idx == mNumber && val > 0.9f;
}



void inferenceExecution(const InferenceOptions& inference, InferenceEnvironment& iEnv, SyncStruct& sync, int offset,
                        int streams, int device, std::vector<InferenceTrace>& trace)
{

    cudaCheck(cudaSetDevice(device));

    IterationStreams iStreams;
    Iteration* iteration = new Iteration(0, inference, *iEnv.context[offset], *iEnv.bindings[offset]);

    iStreams.emplace_back(iteration);

    std::vector<InferenceTrace> localTrace;
    if (!inferenceLoop(iStreams, sync.cpuStart, sync.gpuStart, inference.iterations, 0, 0, localTrace,
                       inference.skipTransfers))
    {
        iEnv.error = true;
    }
    if (!inference.skipTransfers)
    {
        for (auto& s : iStreams)
        {
            s->fetchOutputData();

            //   auto test=  iEnv.bindings[0]->getBindings();
        }
    }

    sync.mutex.lock();
    trace.insert(trace.end(), localTrace.begin(), localTrace.end());
    sync.mutex.unlock();

}



} // namespace

bool runInference(
        const InferenceOptions& inference, InferenceEnvironment& iEnv, int device, std::vector<InferenceTrace>& trace)
{
    trace.resize(0);

    SyncStruct sync;
    sync.sleep = inference.sleep;
    sync.mainStream.sleep(&sync.sleep);
    sync.cpuStart = std::chrono::high_resolution_clock::now();
    sync.gpuStart.record(sync.mainStream);

    inferenceExecution(inference, iEnv, sync, 0, 0, device, trace);

    auto cmpTrace = [](const InferenceTrace& a, const InferenceTrace& b) { return a.h2dStart < b.h2dStart; };
    std::sort(trace.begin(), trace.end(), cmpTrace);
    return !iEnv.error;
}
namespace
{
size_t reportGpuMemory()
{
    static size_t prevFree{0};
    size_t free{0};
    size_t total{0};
    size_t newlyAllocated{0};
    cudaCheck(cudaMemGetInfo(&free, &total));
    sample::gLogInfo << "Free GPU memory = " << free / 1024._MiB;
    if (prevFree != 0)
    {
        newlyAllocated = (prevFree - free);
        sample::gLogInfo << ", newly allocated GPU memory = " << newlyAllocated / 1024._MiB;
    }
    sample::gLogInfo << ", total GPU memory = " << total / 1024._MiB << std::endl;
    prevFree = free;
    return newlyAllocated;
}
} // namespace

//! Returns true if deserialization is slower than expected or fails.
bool timeDeserialize(InferenceEnvironment& iEnv)
{
    TrtUniquePtr<IRuntime> rt{createInferRuntime(sample::gLogger.getTRTLogger())};
    constexpr int32_t kNBMODELS{21};
    TrtUniquePtr<ICudaEngine> engineArray[kNBMODELS];
    TrtUniquePtr<IHostMemory> serializedEngine{iEnv.engine->serialize()};
    size_t free{0};
    size_t total{0};
    cudaCheck(cudaMemGetInfo(&free, &total));

    // Record initial gpu memory state.
    reportGpuMemory();

    sample::gLogInfo << "Begin deserialization engine..." << std::endl;
    auto startClock = std::chrono::high_resolution_clock::now();
    engineArray[0].reset(rt->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size(), nullptr));
    auto endClock = std::chrono::high_resolution_clock::now();
    const float first = std::chrono::duration<float, std::milli>(endClock - startClock).count();
    sample::gLogInfo << "First deserialization time = " << first << " milliseconds" << std::endl;
    const auto engineSizeGpu = reportGpuMemory();

    // Check if first deserialization suceeded.
    if (engineArray[0] == nullptr)
    {
        sample::gLogError << "Engine deserialization failed." << std::endl;
        return true;
    }

    // Normally we want to run 20 iterations, but on smaller GPU cards or larger models, need to restrict
    // the number of iterations so that memory allocations don't fail. We also restrict to 75% of GPU memory
    // so that we reserve memory for the initial model and dependencies.
    const int32_t count = static_cast<int32_t>(std::min<float>(total * .75f / engineSizeGpu, kNBMODELS));
    if (count < 1)
    {
        sample::gLogError << "Not enough memory to deserialize engine a second time." << std::endl;
        // There's no way to determine if if additional serializations will be slower.
        return false;
    }

    startClock = std::chrono::high_resolution_clock::now();
    for (int32_t i = 1; i < count; ++i)
    {
        engineArray[i].reset(rt->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size(), nullptr));
    }
    endClock = std::chrono::high_resolution_clock::now();
    const float totalTime = std::chrono::duration<float, std::milli>(endClock - startClock).count();
    const auto averageTime = totalTime / (count - 1);
    // reportGpuMemory sometimes reports zero after a single deserialization of a small engine,
    // so use the size of memory for all the iterations.
    const auto totalEngineSizeGpu = reportGpuMemory();
    sample::gLogInfo << "Total deserialization time = " << totalTime << " milliseconds, average time = " << averageTime
                     << ", first time = " << first << "." << std::endl;
    sample::gLogInfo << "Deserialization Bandwidth = " << 1E-6 * totalEngineSizeGpu / totalTime << " GB/s" << std::endl;

    // If the first deserialization is more than tolerance slower than
    // the average deserialization, return true, which means an error occurred.
    // See http://nvbugs/2917825 for the motivation behind this condition.
    const auto tolerance = 1.50F;
    const bool isSlowerThanExpected = first > averageTime * tolerance;
    if (isSlowerThanExpected)
    {
        sample::gLogInfo << "First deserialization time divided by average time is " << (first / averageTime)
                         << ". Exceeds tolerance of " << tolerance << "x." << std::endl;
    }
    return isSlowerThanExpected;
}
} // namespace sample
