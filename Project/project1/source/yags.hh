/* @file
 * Header file for YAGS branch predictor
 *
 * 18-640 Foundations of Computer Architecture
 * Carnegie Mellon University
 *
 */

#ifndef __CPU_PRED_YAGS_PRED_HH__
#define __CPU_PRED_YAGS_PRED_HH__

#include "cpu/pred/bpred_unit.hh"
#include "cpu/pred/sat_counter.hh"

/*
 * Feel free to make any modifications, this is a skeleton code
 * to get you started.
 * Note: Do not change name of class
 */
class YagsBP : public BPredUnit
{
  public:
    YagsBP(const Params *params);
    void uncondBranch(void * &bp_history);
    void squash(void *bp_history);
    bool lookup(Addr branch_addr, void * &bp_history);
    void btbUpdate(Addr branch_addr, void * &bp_history);
    void update(Addr branch_addr, bool taken, void *bp_history, bool squashed);
    void retireSquashed(void *bp_history);

  private:
    void updateGlobalHistReg(bool taken);

    struct BPHistory {
        unsigned globalHistoryReg;
        // was the taken array's prediction used?
        // true: takenPred used
        // false: notPred used
        bool takenUsed;
        // prediction of the taken array
        // true: predict taken
        // false: predict not-taken
        bool takenPred;
        // prediction of the not-taken array
        // true: predict taken
        // false: predict not-taken
        bool notTakenPred;
        // the final taken/not-taken prediction
        // true: predict taken
        // false: predict not-taken
        bool finalPred;
        
        bool cacheUsed;
    };

    // choice predictors
    std::vector<SatCounter> choiceCounters;
    // taken direction predictors
    std::vector<SatCounter> takenCounters;
    // not-taken direction predictors
    std::vector<SatCounter> notTakenCounters;
    
    std::vector<unsigned> takenTags;
    
    std::vector<unsigned> notTakenTags;
    unsigned instShiftAmt;

    unsigned globalHistoryReg;
    unsigned globalHistoryBits;
    unsigned historyRegisterMask;

    unsigned choicePredictorSize;
    unsigned choiceCtrBits;
    unsigned choiceHistoryMask;
    unsigned globalPredictorSize;
    unsigned globalCtrBits;
    unsigned globalHistoryMask;

    unsigned choiceThreshold;
    unsigned takenThreshold;
    unsigned notTakenThreshold;
};

#endif // __CPU_PRED_YAGS_PRED_HH__
