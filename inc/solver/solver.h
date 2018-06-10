/*
 * Solver is an abstract class
 *
 */


#ifndef MKT_SOLVER_H
#define MKT_SOLVER_H

#include "net.h"
#include "definitions.h"

namespace mkt {

    template<typename T>
    class KyuNet;

    template<typename T>
    class Solver
    {
    public:
        Solver(KyuNet<T>* Net, T learning_rate) {
            pNet_ = Net;
            learning_rate_ = learning_rate;
        };
        ~Solver(){
            pNet_ = nullptr;
        };

        // Initialize Function
        virtual void initialize()=0;

        // update Function
        virtual void Update()=0;

        //
        void Regularize();

    protected:
        KyuNet<T>* pNet_;
        RegularizationType regularizationType_;
        T learning_rate_;
        T momentum_;
        T lambda_; // L1 or L2 regularization parameter


    };
} // namespace mkt



#endif // SOLVER_H
