/*
 * Solver is an abstract class
 *
 */


#ifndef SOLVER_H
#define SOLVER_H

#include "net.h"
#include "definitions.h"

namespace mkt {

    class KyuNet;

    class Solver
    {
    public:
        Solver(KyuNet* Net) {
            pNet_ = Net;
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
        KyuNet* pNet_;
        RegularizationType regularizationType_;
        float momentum_;
        float lambda_; // L1 or L2 regularization parameter


    };
} // namespace mkt



#endif // SOLVER_H
