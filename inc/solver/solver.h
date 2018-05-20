/*
 * Solver is an abstract class
 *
 */


#ifndef SOLVER_H
#define SOLVER_H

#include "net.h"
#include "definitions.h"

namespace mkt {

    template<typename T>
    class KyuNet;

    template<typename T>
    class Solver
    {
    public:
        Solver(KyuNet<T>* Net) {
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
        KyuNet<T>* pNet_;
        RegularizationType regularizationType_;
        T momentum_;
        T lambda_; // L1 or L2 regularization parameter


    };
} // namespace mkt



#endif // SOLVER_H
