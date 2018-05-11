
#include "solver/sgd_solver.h"

#include "test_utils.hpp"

namespace mkt {

    // Constructor
    SGDSolver::SGDSolver(KyuNet* net): Solver(net) {

        // Initialize Momentum value matrix
        std::vector<Layer*> layers = net->getLayers();
        for (int i = 0; i < layers.size(); ++i)
        {
            if (layers[i]->pW_)
            {
                Shape shape = layers.at(i)->getWeight_Shape();
                // momentums_.emplace_back(new Tensor{shape});
                Tensor* pTensor = new Tensor{shape};
                momentums_.push_back(pTensor);
            } else {
                Tensor* pTensor = new Tensor{1,1,1,1};
                // momentums_.emplace_back(new Tensor{1,1,1,1});
                momentums_.push_back(pTensor);
            }
        }
    }

    // Destructor
    SGDSolver::~SGDSolver(){}

    void SGDSolver::initialize() {

        // allocate Tensor in momentums
        for (int i = 0; i < momentums_.size(); ++i)
        {
            momentums_.at(i)->allocate();
        }
    }


    void SGDSolver::Update() {
        /**
         * Origin SGD:
         * Wt+1 = Wt - alpha * gWt.
         * gWt : gradient of loss function respect to Wt
         *
         *
         * SGD with Momentum:
         * we add previous momentum of gWt to current gWt, so gWt will be
         * gWt_m = (m * gWt-1_m) + (alpha * gWt)
         *          |     |           |      |
         *          |     |           |      |---gradient of loss with respect to weight
         *          |     |           |----------learning rate
         *          |     |----------------------Momentum of previous update value
         *          |----------------------------fraction of momentum
         * Wt+1 = Wt - gWt_m
         */


        // Start from the back of layers
        std::vector<Layer*> layers = pNet_->getLayers();
        for(size_t i = layers.size(); i-- > 0; ) {

            Layer* pLayer = layers.at(i);
            if (pLayer->pW_)
            {
                size_t size = pLayer->pgW_->getWholeSize();
                float* pWData = pLayer->pW_->getCPUData();
                float* pgWData = pLayer->pgW_->getCPUData();

                // gWt_m = momentums[i]
                float* pMomentumData = momentums_.at(i)->getCPUData();



                // y = ax+by
                // cur_update_value = fraction_momentum * pre_Momentum[i] + laerning_rate * pgW_[i]
                axpby(
                    size,           // The size
                    0.0001/64.0f,           // a = learning rate
                    pgWData,        // x
                    0.9,            // b = fraction of momentum
                    pMomentumData   // y
                );

                axpy(size, -1, pMomentumData, pWData);

                // Copy gWt_m back to layer->pgW_
                // we update pW_ = pW_ - pgW_
                mem_copy_cpu(size, pMomentumData, pgWData);
            }
        }
    }
}
