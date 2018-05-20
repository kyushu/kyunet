#ifndef SGD_SOLVER_H
#define SGD_SOLVER_H

#include "solver/solver.h"
#include "tensor.h"


namespace mkt {

    template<typename T>
    class SGDSolver: public Solver<T>
    {
    public:
        SGDSolver(KyuNet<T>* net);
        ~SGDSolver();

        // Initialize Function
        void initialize();


        // update Function
        void Update();

        std::vector<Tensor<T>*> momentums_;
    private:



    };

    template class Solver<float>;

} // namespace mkt


#endif // SGD_SOLVER_H
