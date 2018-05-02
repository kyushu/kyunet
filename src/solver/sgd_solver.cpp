
#include "solver/sgd_solver.h"

namespace mkt {

    // Constructor
    SgdSolver::SgdSolver(KyuNet* net): Solver(net) {

        // Initialize Momentum value matrix
        std::vector<Layer*> layers = net->getLayers();
        for (int i = 0; i < layers.size(); ++i)
        {
            Shape shape = layers.at(i)->getWeightShape();
            Tensor* tensor = new Tensor{shape};
            momentums_.push_back(tensor);
        }

    }

    // Destructor
    SgdSolver::~SgdSolver(){}

    void SgdSolver::initialize() {

        // allocate Tensor in momentums
        for (int i = 0; i < momentums_.size(); ++i)
        {
            momentums_.at(i)->allocate();
        }
    }


    void SgdSolver::Update() {
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
            size_t size = pLayer->pgW_->getWholeSize();
            float* pgWData = pLayer->pgW_->getCPUData();

            // gWt_m = momentums[i]
            float* pMomentumData = momentums_.at(i)->getCPUData();
            axpby(
                size,           // The size
                0.005,           // a = learning rate
                pgWData,        // x
                0.9,            // b = fraction of momentum
                pMomentumData   // y
            );

            // Copy gWt_m back to layer->pgW_
            // we update pW_ = pW_ - pgW_
            mem_copy_cpu(size, pMomentumData, pgWData);
        }

    }
}
