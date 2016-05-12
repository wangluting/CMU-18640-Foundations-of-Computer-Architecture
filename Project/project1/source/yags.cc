/* @file
 * Implementation of a YAGS branch predictor
 *
 * 18-640 Foundations of Computer Architecture
 * Carnegie Mellon University
 *
 */

#include "base/bitfield.hh"
#include "base/intmath.hh"
#include "cpu/pred/yags.hh"


/*
 * Constructor for YagsBP
 */
YagsBP::YagsBP(const Params *params)
    : BPredUnit(params), instShiftAmt(params->instShiftAmt),
      globalHistoryReg(0),
      globalHistoryBits(ceilLog2(params->globalPredictorSize)),
      choicePredictorSize(params->choicePredictorSize),
      choiceCtrBits(params->choiceCtrBits),
      globalPredictorSize(params->globalPredictorSize),
      globalCtrBits(params->globalCtrBits)
{
    if (!isPowerOf2(choicePredictorSize))
        fatal("Invalid choice predictor size.\n");
    if (!isPowerOf2(globalPredictorSize))
        fatal("Invalid global history predictor size.\n");
    choiceCounters.resize(choicePredictorSize);
    takenCounters.resize(globalPredictorSize);
    notTakenCounters.resize(globalPredictorSize);
    takenTags.resize(globalPredictorSize);
    notTakenTags.resize(globalPredictorSize);
    for (int i = 0; i < choicePredictorSize; ++i) {
        choiceCounters[i].setBits(choiceCtrBits);
    }
    for (int i = 0; i < globalPredictorSize; ++i) {
        takenCounters[i].setBits(globalCtrBits);
        notTakenCounters[i].setBits(globalCtrBits);
        //takenTags[i]=0;
        //notTakenTags[i]=0;
    }
    historyRegisterMask = mask(globalHistoryBits);
    choiceHistoryMask = choicePredictorSize - 1;
    globalHistoryMask = globalPredictorSize - 1;

    choiceThreshold = (ULL(1) << (choiceCtrBits - 1)) - 1;
    takenThreshold = (ULL(1) << (choiceCtrBits - 1)) - 1;
    notTakenThreshold = (ULL(1) << (choiceCtrBits - 1)) - 1;
}

/*
 * Actions for an unconditional branch
 */
void
YagsBP::uncondBranch(void * &bpHistory)
{
    BPHistory *history = new BPHistory;
    history->globalHistoryReg = globalHistoryReg;
    history->takenUsed = true;
    history->takenPred = true;
    history->notTakenPred = true;
    history->finalPred = true;
    history->cacheUsed = false;
    bpHistory = static_cast<void*>(history);
    updateGlobalHistReg(true);
}

/*
 * Actions for squash
 */
void
YagsBP::squash(void *bpHistory)
{
    BPHistory *history = static_cast<BPHistory*>(bpHistory);
    globalHistoryReg = history->globalHistoryReg;

    delete history;
}

/*
 * Lookup the actual branch prediction.
 */
bool
YagsBP::lookup(Addr branchAddr, void * &bpHistory)
{
    unsigned tagMask = (ULL(1) << 8) - 1;
    unsigned choiceHistoryIdx = ((branchAddr >> instShiftAmt)
                                & choiceHistoryMask);
    unsigned globalHistoryIdx = (((branchAddr >> instShiftAmt)
                                ^ globalHistoryReg)
                                & globalHistoryMask);

    unsigned leastSigAddr = branchAddr & tagMask;
    unsigned tagsIdx = globalHistoryIdx;
    assert(choiceHistoryIdx < choicePredictorSize);
    assert(globalHistoryIdx < globalPredictorSize);

    bool choicePrediction = choiceCounters[choiceHistoryIdx].read()
                            > choiceThreshold;
    bool takenGHBPrediction = takenCounters[globalHistoryIdx].read()
                              > takenThreshold;
    bool notTakenGHBPrediction = notTakenCounters[globalHistoryIdx].read()
                                 > notTakenThreshold;
    
    
    bool cacheUsed  = false;
    bool finalPrediction;
    BPHistory *history = new BPHistory;

    history->globalHistoryReg = globalHistoryReg;
    history->takenUsed = choicePrediction;

    history->takenPred = takenGHBPrediction;
    history->notTakenPred = notTakenGHBPrediction;
    
    
    if (choicePrediction) {
        //fatal("Invalid choice predictor size.\n");
        if(notTakenTags[tagsIdx]!=leastSigAddr){

        
        finalPrediction = choicePrediction;
       
        }else {     
          
                 finalPrediction = notTakenGHBPrediction;
                 cacheUsed = true;
          
          }
    }
     else {
        if(takenTags[tagsIdx] !=leastSigAddr){
      //takenTags[tagsIdx] = leastSigAddr;
      finalPrediction = choicePrediction;

    }else {
        //if(takenTags[tagsIdx] == leastSigAddr)
            finalPrediction = takenGHBPrediction;
            cacheUsed  = true;
        //else 
          //  finalPrediction = choicePrediction;
       }
    }
    history->finalPred = finalPrediction;
    history->cacheUsed = cacheUsed;
    bpHistory = static_cast<void*>(history);
    updateGlobalHistReg(finalPrediction);

    return finalPrediction;
}

/*
 * BTB Update actions
 */
void
YagsBP::btbUpdate(Addr branchAddr, void * &bpHistory)
{
    globalHistoryReg &= (historyRegisterMask & ~ULL(1));
}

/*
 * Update data structures after getting actual decison
 */
void
YagsBP::update(Addr branchAddr, bool taken, void *bpHistory, bool squashed)
{ 

if (bpHistory) {
    unsigned tagMask = (ULL(1) << 8) - 1;
        BPHistory *history = static_cast<BPHistory*>(bpHistory);
        unsigned choiceHistoryIdx = ((branchAddr >> instShiftAmt)
                                    & choiceHistoryMask);
        unsigned globalHistoryIdx = (((branchAddr >> instShiftAmt)
                                    ^ history->globalHistoryReg)
                                    & globalHistoryMask);

        unsigned leastSigAddr = branchAddr & tagMask;

        assert(choiceHistoryIdx < choicePredictorSize);
        assert(globalHistoryIdx < globalPredictorSize);

        if(history->cacheUsed){
            if (history->takenUsed) {
                if (taken) {
                    notTakenCounters[globalHistoryIdx].increment();
                } else {
                    notTakenCounters[globalHistoryIdx].decrement();
                }
            } else {
                if (taken) {
                    takenCounters[globalHistoryIdx].increment();
                } else {
                    takenCounters[globalHistoryIdx].decrement();
                }
            }
        }
        else{
            if(history->finalPred!=taken){
                if(taken){
                    takenTags[globalHistoryIdx]=leastSigAddr;
                }
                else{
                    notTakenTags[globalHistoryIdx]=leastSigAddr;
                }
            }
        }

   

        if(history->cacheUsed){
        if (history->finalPred == taken) {
            
            if (taken == history->takenUsed) {
                if (taken) {
                    choiceCounters[choiceHistoryIdx].increment();
                } else {
                    choiceCounters[choiceHistoryIdx].decrement();
                }
            } 
        } else {
            
            if (taken) {
                choiceCounters[choiceHistoryIdx].increment();

          
            } else {
                choiceCounters[choiceHistoryIdx].decrement();
           
            }
        }
    }else{

        if(taken){
                choiceCounters[choiceHistoryIdx].increment();            
        }else {
                choiceCounters[choiceHistoryIdx].decrement();
        }

    }
        if (squashed) {
            if (taken) {
                globalHistoryReg = (history->globalHistoryReg << 1) | 1;
            } else {
                globalHistoryReg = (history->globalHistoryReg << 1);
            }
            globalHistoryReg &= historyRegisterMask;
        } else {
            delete history;
        }
    }

}

/*
 * Retire Squashed Instruction
 */
void
YagsBP::retireSquashed(void *bp_history)
{
    BPHistory *history = static_cast<BPHistory*>(bp_history);
    delete history;
}

/*
 * Global History Registor Update
 */
void
YagsBP::updateGlobalHistReg(bool taken)
{
    globalHistoryReg = taken ? (globalHistoryReg << 1) | 1 :
                               (globalHistoryReg << 1);
    globalHistoryReg &= historyRegisterMask;
}
