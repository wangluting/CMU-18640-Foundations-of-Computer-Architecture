/* @file
 * Implementation of a Gshare branch predictor
 *
 * 18-640 Foundations of Computer Architecture
 * Carnegie Mellon University
 *
 */

#include "base/bitfield.hh"
#include "base/intmath.hh"
#include "cpu/pred/gshare.hh"

/*
 * Constructor for GshareBP
 */
GshareBP::GshareBP(const Params *params)
    : BPredUnit(params), instShiftAmt(params->instShiftAmt),
      globalHistoryReg(0),
      globalHistoryBits(ceilLog2(params->localPredictorSize)),
      localPredictorSize(params->localPredictorSize),
      localCtrBits(params->localCtrBits)
{
    if (!isPowerOf2(localPredictorSize))
        fatal("Invalid local predictor size.\n");

    localCtrs.resize(localPredictorSize);

    for (int i = 0; i < localPredictorSize; ++i) {
        localCtrs[i].setBits(localCtrBits);
    }

    historyRegisterMask = mask(globalHistoryBits);
    localPredictorMask = localPredictorSize -1;

    localThreshold = (ULL(1) << (localCtrBits - 1)) - 1;
}
/*
* Reset Data Structures
*/
void
GshareBP::reset()
{
    for (int i = 0; i < localPredictorSize; ++i) {
        localCtrs[i].reset();
    }
    //globalHistoryReg = 0;
}


/*
 * Actions for an unconditional branch
 */
void
GshareBP::uncondBranch(void * &bpHistory)
{
    BPHistory *history = new BPHistory;
    history->globalHistoryReg = globalHistoryReg;
    history->finalPred = true;
    bpHistory = static_cast<void*>(history);
    updateGlobalHistReg(true);
}

/*
 * Lookup the actual branch prediction.
 */

bool
GshareBP::lookup(Addr branchAddr, void * &bpHistory)
{
    unsigned localPredictorIdx = (((branchAddr >> instShiftAmt)//XOR
                                ^ globalHistoryReg)
                                & localPredictorMask);

    assert(localPredictorIdx < localPredictorSize );

    bool finalPrediction = localCtrs[localPredictorIdx].read()
                            > localThreshold;

    BPHistory *history = new BPHistory;
    history->globalHistoryReg = globalHistoryReg;

    history->finalPred = finalPrediction;


    bpHistory = static_cast<void*>(history);
    updateGlobalHistReg(finalPrediction);

    return finalPrediction;
}

/*
 * BTB Update actions
 */
void
GshareBP::btbUpdate(Addr branchAddr, void * &bpHistory)
{
  globalHistoryReg &= (historyRegisterMask & ~ULL(1));
}

/*
 * Update data structures after getting actual decison 
 */
void
GshareBP::update(Addr branchAddr, bool taken, void *bpHistory, bool squashed)
{
   if (bpHistory) {
        BPHistory *history = static_cast<BPHistory*>(bpHistory);

        unsigned localPredictorIdx = (((branchAddr >> instShiftAmt)
        			     ^ history->globalHistoryReg)
                                     & localPredictorMask);

        assert(localPredictorIdx < localPredictorSize);

        if (taken)
        {
         localCtrs[localPredictorIdx].increment();
        } else {
         localCtrs[localPredictorIdx].decrement();
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
 * Global History Registor Update 
 */
void
GshareBP::updateGlobalHistReg(bool taken)
{
    globalHistoryReg = taken ? (globalHistoryReg << 1) | 1 :
                               (globalHistoryReg << 1);
    globalHistoryReg &= historyRegisterMask;
}

/*
 * Actions for squash
 */
void
GshareBP::squash(void *bpHistory) {
    BPHistory *history = static_cast<BPHistory*>(bpHistory);
    globalHistoryReg = history->globalHistoryReg;

    delete history;
}
