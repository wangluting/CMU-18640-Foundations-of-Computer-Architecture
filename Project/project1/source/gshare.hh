/* @file
 * Header file for Gshare branch predictor
 * 
 * 18-640 Foundations of Computer Architecture
 * Carnegie Mellon University
 *
 */

#ifndef __CPU_PRED_GSHARE_PRED_HH__
#define __CPU_PRED_GSHARE_PRED_HH__

#include "cpu/pred/bpred_unit.hh"
#include "cpu/pred/sat_counter.hh"

/*
 * Feel free to make any modifications, this is a skeleton code
 * to get you started.
 * Note: Do not change name of class
 */
class GshareBP : public BPredUnit
{
  public:
    GshareBP(const Params *params);
    void uncondBranch(void * &bp_history);
    void squash(void *bp_history);
    bool lookup(Addr branch_addr, void * &bp_history);
    void btbUpdate(Addr branch_addr, void * &bp_history);
    void update(Addr branch_addr, bool taken, void *bp_history, bool squashed);
    void reset();

  private:
    void updateGlobalHistReg(bool taken);

    struct BPHistory {
        unsigned globalHistoryReg;
        bool finalPred;
    };

    unsigned instShiftAmt;

    unsigned globalHistoryReg;
    unsigned globalHistoryBits;
    unsigned historyRegisterMask;

    /** Local counters. */
    std::vector<SatCounter> localCtrs;
    /** Number of counters in the local predictor. */
    unsigned localPredictorSize;
    /** Number of bits of the local predictor's counters. */
    unsigned localCtrBits;

    unsigned localPredictorMask;

    unsigned localThreshold;
};

#endif // __CPU_PRED_GSHARE_PRED_HH__
