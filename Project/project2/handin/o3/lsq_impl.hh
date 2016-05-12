/*
 * Copyright (c) 2011-2012 ARM Limited
 * All rights reserved
 *
 * The license below extends only to copyright in the software and shall
 * not be construed as granting a license to any other intellectual
 * property including but not limited to intellectual property relating
 * to a hardware implementation of the functionality of the software
 * licensed hereunder.  You may use the software subject to the license
 * terms below provided that you ensure that this notice is replicated
 * unmodified and in its entirety in all distributions of the software,
 * modified or unmodified, in source code or in binary form.
 *
 * Copyright (c) 2005-2006 The Regents of The University of Michigan
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Korey Sewell
 */

#ifndef __CPU_O3_LSQ_IMPL_HH__
#define __CPU_O3_LSQ_IMPL_HH__

#include <algorithm>
#include <list>
#include <string>

#include "cpu/o3/lsq.hh"
#include "debug/Drain.hh"
#include "debug/Fetch.hh"
#include "debug/LSQ.hh"
#include "debug/Writeback.hh"
#include "params/DerivO3CPU.hh"

using namespace std;

template <class Impl>
LSQ<Impl>::LSQ(O3CPU *cpu_ptr, IEW *iew_ptr, DerivO3CPUParams *params)
    : cpu(cpu_ptr), iewStage(iew_ptr),
      LQEntries(params->LQEntries),
      SQEntries(params->SQEntries),
      numThreads(params->numThreads),
      retryTid(-1)
{
    assert(numThreads > 0 && numThreads <= Impl::MaxThreads);

    //**********************************************/
    //************ Handle SMT Parameters ***********/
    //**********************************************/
    std::string policy = params->smtLSQPolicy;

    //Convert string to lowercase
    std::transform(policy.begin(), policy.end(), policy.begin(),
                   (int(*)(int)) tolower);

    //Figure out fetch policy
    if (policy == "dynamic") {
        lsqPolicy = Dynamic;

        maxLQEntries = LQEntries;
        maxSQEntries = SQEntries;

        DPRINTF(LSQ, "LSQ sharing policy set to Dynamic\n");
    } else if (policy == "partitioned") {
        lsqPolicy = Partitioned;

        //@todo:make work if part_amt doesnt divide evenly.
        maxLQEntries = LQEntries / numThreads;
        maxSQEntries = SQEntries / numThreads;

        DPRINTF(Fetch, "LSQ sharing policy set to Partitioned: "
                "%i entries per LQ | %i entries per SQ\n",
                maxLQEntries,maxSQEntries);
    } else if (policy == "threshold") {
        lsqPolicy = Threshold;

        assert(params->smtLSQThreshold > LQEntries);
        assert(params->smtLSQThreshold > SQEntries);

        //Divide up by threshold amount
        //@todo: Should threads check the max and the total
        //amount of the LSQ
        maxLQEntries  = params->smtLSQThreshold;
        maxSQEntries  = params->smtLSQThreshold;

        DPRINTF(LSQ, "LSQ sharing policy set to Threshold: "
                "%i entries per LQ | %i entries per SQ\n",
                maxLQEntries,maxSQEntries);
    } else {
        assert(0 && "Invalid LSQ Sharing Policy.Options Are:{Dynamic,"
                    "Partitioned, Threshold}");
    }

    //Initialize LSQs
    thread = new LSQUnit[numThreads];
    for (ThreadID tid = 0; tid < numThreads; tid++) {
        thread[tid].init(cpu, iew_ptr, params, this,
                         maxLQEntries, maxSQEntries, tid);
        thread[tid].setDcachePort(&cpu_ptr->getDataPort());
    }
}


template<class Impl>
std::string
LSQ<Impl>::name() const
{
    return iewStage->name() + ".lsq";
}

template<class Impl>
void
LSQ<Impl>::regStats()
{
    //Initialize LSQs
    for (ThreadID tid = 0; tid < numThreads; tid++) {
        thread[tid].regStats();
    }
	
	//--------luting-------
	LoadQueueB1
        .name("LoadQueueB1")
        .desc("Number of cycles in B1 of Load Queue");

    LoadQueueB2
        .name("LoadQueueB2")
        .desc("Number of cycles in B1 of Load Queue");

    LoadQueueB3
        .name("LoadQueueB3")
        .desc("Number of cycles in B1 of Load Queue");


    LoadQueueB4
        .name("LoadQueueB4")
        .desc("Number of cycles in B1 of Load Queue");


    StoreQueueB1
        .name("StoreQueueB1")
        .desc("Number of cycles in B1 of Store Queue");

    StoreQueueB2
        .name("StoreQueueB2")
        .desc("Number of cycles in B1 of Store Queue");

    StoreQueueB3
        .name("StoreQueueB3")
        .desc("Number of cycles in B1 of Store Queue");

    StoreQueueB4
        .name("StoreQueueB4")
        .desc("Number of cycles in B1 of Store Queue");
}

template<class Impl>
void
LSQ<Impl>::setActiveThreads(list<ThreadID> *at_ptr)
{
    activeThreads = at_ptr;
    assert(activeThreads != 0);
}

template <class Impl>
void
LSQ<Impl>::drainSanityCheck() const
{
    assert(isDrained());

    for (ThreadID tid = 0; tid < numThreads; tid++)
        thread[tid].drainSanityCheck();
}

template <class Impl>
bool
LSQ<Impl>::isDrained() const
{
    bool drained(true);

    if (!lqEmpty()) {
        DPRINTF(Drain, "Not drained, LQ not empty.\n");
        drained = false;
    }

    if (!sqEmpty()) {
        DPRINTF(Drain, "Not drained, SQ not empty.\n");
        drained = false;
    }

    if (retryTid != InvalidThreadID) {
        DPRINTF(Drain, "Not drained, the LSQ has blocked the caches.\n");
        drained = false;
    }

    return drained;
}

template <class Impl>
void
LSQ<Impl>::takeOverFrom()
{
    for (ThreadID tid = 0; tid < numThreads; tid++) {
        thread[tid].takeOverFrom();
    }
}

template <class Impl>
int
LSQ<Impl>::entryAmount(ThreadID num_threads)
{
    if (lsqPolicy == Partitioned) {
        return LQEntries / num_threads;
    } else {
        return 0;
    }
}

template <class Impl>
void
LSQ<Impl>::resetEntries()
{
    if (lsqPolicy != Dynamic || numThreads > 1) {
        int active_threads = activeThreads->size();

        int maxEntries;

        if (lsqPolicy == Partitioned) {
            maxEntries = LQEntries / active_threads;
        } else if (lsqPolicy == Threshold && active_threads == 1) {
            maxEntries = LQEntries;
        } else {
            maxEntries = LQEntries;
        }

        list<ThreadID>::iterator threads  = activeThreads->begin();
        list<ThreadID>::iterator end = activeThreads->end();

        while (threads != end) {
            ThreadID tid = *threads++;

            resizeEntries(maxEntries, tid);
        }
    }
}

template<class Impl>
void
LSQ<Impl>::removeEntries(ThreadID tid)
{
    thread[tid].clearLQ();
    thread[tid].clearSQ();
}

template<class Impl>
void
LSQ<Impl>::resizeEntries(unsigned size, ThreadID tid)
{
    thread[tid].resizeLQ(size);
    thread[tid].resizeSQ(size);
}

template<class Impl>
void
LSQ<Impl>::tick()
{
    list<ThreadID>::iterator threads = activeThreads->begin();
    list<ThreadID>::iterator end = activeThreads->end();

    while (threads != end) {
        ThreadID tid = *threads++;

        thread[tid].tick();
    }
}

template<class Impl>
void
LSQ<Impl>::insertLoad(DynInstPtr &load_inst)
{
    ThreadID tid = load_inst->threadNumber;

    thread[tid].insertLoad(load_inst);
	

}

template<class Impl>
void
LSQ<Impl>::insertStore(DynInstPtr &store_inst)
{
    ThreadID tid = store_inst->threadNumber;

    thread[tid].insertStore(store_inst);
	

	

}

template<class Impl>
Fault
LSQ<Impl>::executeLoad(DynInstPtr &inst)
{
    ThreadID tid = inst->threadNumber;

    return thread[tid].executeLoad(inst);
}

template<class Impl>
Fault
LSQ<Impl>::executeStore(DynInstPtr &inst)
{
    ThreadID tid = inst->threadNumber;

    return thread[tid].executeStore(inst);
}

template<class Impl>
void
LSQ<Impl>::writebackStores()
{
    list<ThreadID>::iterator threads = activeThreads->begin();
    list<ThreadID>::iterator end = activeThreads->end();

    while (threads != end) {
        ThreadID tid = *threads++;

        if (numStoresToWB(tid) > 0) {
            DPRINTF(Writeback,"[tid:%i] Writing back stores. %i stores "
                "available for Writeback.\n", tid, numStoresToWB(tid));
        }

        thread[tid].writebackStores();
    }
}

template<class Impl>
bool
LSQ<Impl>::violation()
{
    /* Answers: Does Anybody Have a Violation?*/
    list<ThreadID>::iterator threads = activeThreads->begin();
    list<ThreadID>::iterator end = activeThreads->end();

    while (threads != end) {
        ThreadID tid = *threads++;

        if (thread[tid].violation())
            return true;
    }

    return false;
}

template <class Impl>
void
LSQ<Impl>::recvRetry()
{
    if (retryTid == InvalidThreadID)
    {
        //Squashed, so drop it
        return;
    }
    int curr_retry_tid = retryTid;
    // Speculatively clear the retry Tid.  This will get set again if
    // the LSQUnit was unable to complete its access.
    retryTid = -1;
    thread[curr_retry_tid].recvRetry();
}

template <class Impl>
bool
LSQ<Impl>::recvTimingResp(PacketPtr pkt)
{
    if (pkt->isError())
        DPRINTF(LSQ, "Got error packet back for address: %#X\n",
                pkt->getAddr());
    thread[pkt->req->threadId()].completeDataAccess(pkt);
    return true;
}

template <class Impl>
void
LSQ<Impl>::recvTimingSnoopReq(PacketPtr pkt)
{
    DPRINTF(LSQ, "received pkt for addr:%#x %s\n", pkt->getAddr(),
            pkt->cmdString());

    // must be a snoop
    if (pkt->isInvalidate()) {
        DPRINTF(LSQ, "received invalidation for addr:%#x\n",
                pkt->getAddr());
        for (ThreadID tid = 0; tid < numThreads; tid++) {
            thread[tid].checkSnoop(pkt);
        }
    }
}

template<class Impl>
int
LSQ<Impl>::getCount()
{
    unsigned total = 0;

    list<ThreadID>::iterator threads = activeThreads->begin();
    list<ThreadID>::iterator end = activeThreads->end();

    while (threads != end) {
        ThreadID tid = *threads++;

        total += getCount(tid);
    }

    return total;
}

template<class Impl>
int
LSQ<Impl>::numLoads()
{
    unsigned total = 0;

    list<ThreadID>::iterator threads = activeThreads->begin();
    list<ThreadID>::iterator end = activeThreads->end();

    while (threads != end) {
        ThreadID tid = *threads++;

        total += numLoads(tid);
    }

    return total;
}

template<class Impl>
int
LSQ<Impl>::numStores()
{
    unsigned total = 0;

    list<ThreadID>::iterator threads = activeThreads->begin();
    list<ThreadID>::iterator end = activeThreads->end();

    while (threads != end) {
        ThreadID tid = *threads++;

        total += thread[tid].numStores();
    }

    return total;
}

template<class Impl>
unsigned
LSQ<Impl>::numFreeEntries()
{
    unsigned total = 0;

    list<ThreadID>::iterator threads = activeThreads->begin();
    list<ThreadID>::iterator end = activeThreads->end();

    while (threads != end) {
        ThreadID tid = *threads++;

        total += thread[tid].numFreeEntries();
    }

    return total;
}


//---------luting-----------
template<class Impl>
void
LSQ<Impl>::updateLoadQueueCounter(){

	int numberOfUsedBuffer = numLoads();
    float utilization = numberOfUsedBuffer*1.0/32;
    if(utilization < 0.25) LoadQueueB1++;
    else if(utilization < 0.50 ) LoadQueueB2++;
    else if(utilization < 0.75 ) LoadQueueB3++;
    else LoadQueueB4++;
}

template<class Impl>
void
LSQ<Impl>::updateStoreQueueCounter(){

	int numberOfUsedBuffer = numStores();
    float utilization = numberOfUsedBuffer*1.0/32;
    if(utilization < 0.25) StoreQueueB1++;
    else if(utilization < 0.50 ) StoreQueueB2++;
    else if(utilization < 0.75 ) StoreQueueB3++;
    else StoreQueueB4++;
}


template<class Impl>
unsigned
LSQ<Impl>::numFreeEntries(ThreadID tid)
{
    //if (lsqPolicy == Dynamic)
    //return numFreeEntries();
    //else
        return thread[tid].numFreeEntries();
}

template<class Impl>
bool
LSQ<Impl>::isFull()
{
    list<ThreadID>::iterator threads = activeThreads->begin();
    list<ThreadID>::iterator end = activeThreads->end();

    while (threads != end) {
        ThreadID tid = *threads++;

        if (!(thread[tid].lqFull() || thread[tid].sqFull()))
            return false;
    }

    return true;
}

template<class Impl>
bool
LSQ<Impl>::isFull(ThreadID tid)
{
    //@todo: Change to Calculate All Entries for
    //Dynamic Policy
    if (lsqPolicy == Dynamic)
        return isFull();
    else
        return thread[tid].lqFull() || thread[tid].sqFull();
}

template<class Impl>
bool
LSQ<Impl>::isEmpty() const
{
    return lqEmpty() && sqEmpty();
}

template<class Impl>
bool
LSQ<Impl>::lqEmpty() const
{
    list<ThreadID>::const_iterator threads = activeThreads->begin();
    list<ThreadID>::const_iterator end = activeThreads->end();

    while (threads != end) {
        ThreadID tid = *threads++;

        if (!thread[tid].lqEmpty())
            return false;
    }

    return true;
}

template<class Impl>
bool
LSQ<Impl>::sqEmpty() const
{
    list<ThreadID>::const_iterator threads = activeThreads->begin();
    list<ThreadID>::const_iterator end = activeThreads->end();

    while (threads != end) {
        ThreadID tid = *threads++;

        if (!thread[tid].sqEmpty())
            return false;
    }

    return true;
}

template<class Impl>
bool
LSQ<Impl>::lqFull()
{
    list<ThreadID>::iterator threads = activeThreads->begin();
    list<ThreadID>::iterator end = activeThreads->end();

    while (threads != end) {
        ThreadID tid = *threads++;

        if (!thread[tid].lqFull())
            return false;
    }

    return true;
}

template<class Impl>
bool
LSQ<Impl>::lqFull(ThreadID tid)
{
    //@todo: Change to Calculate All Entries for
    //Dynamic Policy
    if (lsqPolicy == Dynamic)
        return lqFull();
    else
        return thread[tid].lqFull();
}

template<class Impl>
bool
LSQ<Impl>::sqFull()
{
    list<ThreadID>::iterator threads = activeThreads->begin();
    list<ThreadID>::iterator end = activeThreads->end();

    while (threads != end) {
        ThreadID tid = *threads++;

        if (!sqFull(tid))
            return false;
    }

    return true;
}

template<class Impl>
bool
LSQ<Impl>::sqFull(ThreadID tid)
{
     //@todo: Change to Calculate All Entries for
    //Dynamic Policy
    if (lsqPolicy == Dynamic)
        return sqFull();
    else
        return thread[tid].sqFull();
}

template<class Impl>
bool
LSQ<Impl>::isStalled()
{
    list<ThreadID>::iterator threads = activeThreads->begin();
    list<ThreadID>::iterator end = activeThreads->end();

    while (threads != end) {
        ThreadID tid = *threads++;

        if (!thread[tid].isStalled())
            return false;
    }

    return true;
}

template<class Impl>
bool
LSQ<Impl>::isStalled(ThreadID tid)
{
    if (lsqPolicy == Dynamic)
        return isStalled();
    else
        return thread[tid].isStalled();
}

template<class Impl>
bool
LSQ<Impl>::hasStoresToWB()
{
    list<ThreadID>::iterator threads = activeThreads->begin();
    list<ThreadID>::iterator end = activeThreads->end();

    while (threads != end) {
        ThreadID tid = *threads++;

        if (hasStoresToWB(tid))
            return true;
    }

    return false;
}

template<class Impl>
bool
LSQ<Impl>::willWB()
{
    list<ThreadID>::iterator threads = activeThreads->begin();
    list<ThreadID>::iterator end = activeThreads->end();

    while (threads != end) {
        ThreadID tid = *threads++;

        if (willWB(tid))
            return true;
    }

    return false;
}

template<class Impl>
void
LSQ<Impl>::dumpInsts() const
{
    list<ThreadID>::const_iterator threads = activeThreads->begin();
    list<ThreadID>::const_iterator end = activeThreads->end();

    while (threads != end) {
        ThreadID tid = *threads++;

        thread[tid].dumpInsts();
    }
}

#endif//__CPU_O3_LSQ_IMPL_HH__