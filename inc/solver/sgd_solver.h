#ifndef SGD_SOLVER_H
#define SGD_SOLVER_H

#include "solver/solver.h"
#include "tensor.h"


namespace mkt {

    class SGDSolver: public Solver
    {
    public:
        SGDSolver(KyuNet* net);
        ~SGDSolver();

        // Initialize Function
        void initialize();


        // update Function
        void Update();

        std::vector<Tensor*> momentums_;
    private:



    };

} // namespace mkt


#endif // SGD_SOLVER_H
