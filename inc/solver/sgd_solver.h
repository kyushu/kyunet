#ifndef SGD_SOLVER_H
#define SGD_SOLVER_H

#include "solver/solver.h"
#include "tensor.h"


namespace mkt {

    class SgdSolver: public Solver
    {
    public:
        SgdSolver(KyuNet* net);
        ~SgdSolver();

        // Initialize Function
        void initialize();


        // update Function
        void Update();

    private:
        std::vector<Tensor*> momentums_;


    };

} // namespace mkt


#endif // SGD_SOLVER_H
