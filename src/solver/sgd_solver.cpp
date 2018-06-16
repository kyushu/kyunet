
#include "solver/sgd_solver.h"


namespace mkt {

    // Constructor
    template<typename T>
    SGDSolver<T>::SGDSolver(KyuNet<T>* net, int batchSize, T learning_rate): Solver<T>(net, learning_rate) {

        this->batchSize_ = batchSize;

        // Initialize Momentum value matrix
        std::vector<Layer<T>*> layers = net->getLayers();
        for (int i = 0; i < layers.size(); ++i)
        {
            if (layers[i]->pW_)
            {
                Shape shape = layers.at(i)->getWeight_Shape();
                momentums_.emplace_back(new Tensor<T>{shape});
                // Tensor<T>* pTensor = new Tensor<T>{shape};
                // momentums_.push_back(pTensor);
            } else {
                momentums_.emplace_back(new Tensor<T>{1,1,1,1});
                // Tensor<T>* pTensor = new Tensor<T>{1,1,1,1};
                // momentums_.push_back(pTensor);
            }
        }
    }

    // Destructor
    template<typename T>
    SGDSolver<T>::~SGDSolver(){}

    template<typename T>
    void SGDSolver<T>::initialize() {

        // allocate Tensor in momentums
        for (int i = 0; i < momentums_.size(); ++i)
        {
            momentums_.at(i)->allocate();
        }
    }

    template<typename T>
    void SGDSolver<T>::Update() {
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
         *          |     |           |      |___ gradient of loss with respect to weight
         *          |     |           |__________ learning rate
         *          |     |______________________ Momentum of previous update value
         *          |____________________________ fraction of momentum
         * Wt+1 = Wt - gWt_m
         */


        // Start from the back of layers
        std::vector<Layer<T>*> layers = this->pNet_->getLayers();
        T learning_rate = this->learning_rate_ / this->batchSize_;
        for(size_t i = layers.size(); i-- > 0; ) {

            Layer<T>* pLayer = layers.at(i);
            if (pLayer->pW_)
            {
                size_t size = pLayer->pgW_->getWholeSize();
                T* pWData = pLayer->pW_->getCPUData();
                T* pgWData = pLayer->pgW_->getCPUData();

                // gWt_m = momentums[i]
                T* pMomentumData = momentums_.at(i)->getCPUData();



                // y = ax+by
                // cur_update_value = fraction_momentum * pre_Momentum[i] + laerning_rate * pgW_[i]
                axpby(
                    size,            // The size
                    // 0.0001/64.0f, // a = learning rate
                    learning_rate,   // a = learning rate
                    pgWData,         // x
                    0.9,             // b = fraction of momentum
                    pMomentumData    // y
                );

                axpy(size, -1, pMomentumData, pWData);

                // Copy gWt_m back to layer->pgW_
                // we update pW_ = pW_ - pgW_
                mem_copy_cpu(size, pMomentumData, pgWData);
            }
        }
    }

    // Explicitly instantiate the template, and its member definitions
    template class SGDSolver<float>;

} // namespace mkt
