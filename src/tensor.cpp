
#include "tensor.h"
#include "inc_thirdparty.h"



namespace mkt {

    /**
     * Constructor without parameters
     */
    template<typename T>
    Tensor<T>::Tensor():
        num_{0},
        channel_{0},
        height_{0},
        width_{0},
        wrCount_{0},
        size2D_{0},
        size3D_{0},
        wholeSize_{0},
        pData_{nullptr}
    {};

    /**
     * Constructor with 4 Dimension parameters
     */
    template<typename T>
    Tensor<T>::Tensor(int num, int ch, int height, int width):
        num_{num},
        channel_{ch},
        height_{height},
        width_{width}
        
    {
        wrCount_ = 0;

        size2D_ = height_ * width_;
        size3D_ = size2D_ * channel_;
        wholeSize_ = num_ * size3D_;
        // fprintf(stderr, "tensor construct num_: %d\n", num_);
        // fprintf(stderr, "tensor construct height_: %d\n", height);
        // fprintf(stderr, "tensor construct width_: %d\n", width_);
        // fprintf(stderr, "tensor construct channel_: %d\n", channel_);
    };

    /**
     * Constructor with Shape
     */
    template<typename T>
    Tensor<T>::Tensor(Shape shape) 
    {
        wrCount_ = 0;

        num_ = shape[0];
        channel_ = shape[1];
        height_ = shape[2];
        width_ = shape[3];


        size2D_ = height_ * width_;
        size3D_ = size2D_ * channel_;
        wholeSize_ = num_ * size3D_;
    }

    // Destructor
    template<typename T>
    Tensor<T>::~Tensor(){
        delete[] pData_;
    };

    // Init Tensor: allocate memory space of pData

    /*************
     * Initialize
     *************/
    template<typename T>
    void Tensor<T>::allocate() {

        // fprintf(stderr, "init_type: %d\n", init_type);
        // size2D_ = height_ * width_;
        // size3D_ = size2D_ * channel_;
        // wholeSize_ = num_ * size3D_;

        if (wholeSize_ == 0)
        {
            fprintf(stderr, "wholeSize == 0\n");
            return;
        }

        pData_ = new T[wholeSize_];

    }

    /**
     * There are two ways to reshape tensor
     * 1. If new shape is smaller then current, we don't re-allocate
     *      the memory space but just update the domension parameters.
     *      In this way we may save the time cost of allocate, release
     *      memory space but waste some memory space.
     * 2. Release current memory space and allocate new memory space with
     *      new dimension parameters.
     *      In this way we keep the memory size as required but spend more
     *      time for re-alocate memory space.
     *
     * The DEFAULT of reAllocate = TRUE
     */
    template<typename T>
    void Tensor<T>::Reshape(int num, int ch, int height, int width, bool reAllocate) {

        int ori_whileSize = wholeSize_;

        // Update dimension parameters
        num_ = num;
        height_ = height;
        width_ = width;
        channel_ = ch;

        size2D_ = height_ * width_;
        size3D_ = size2D_ * channel_;
        wholeSize_ = num_ * size3D_;

        //
        if (reAllocate)
        {
            if (pData_)
            {
                delete[] pData_;
            }
            allocate();
        }
        else {
            if (pData_)
            {
                if (ori_whileSize < wholeSize_)
                {
                    delete[] pData_;
                    allocate();
                }
            } else {
                allocate();
            }
        }
    }

    /**
     * add data from file
     */

    /**
     * add data from memory chunk
     */
    template<typename T>
    OP_STATUS Tensor<T>::addData(const T *pImg, int size) {

        assert(pImg);

        // Safety Check
        if (size != wholeSize_)
        {
            return OP_STATUS::UNMATCHED_SIZE;
        }

        // Get current write address
        T* ptr = pData_;

        for (int i = 0; i < wholeSize_; ++i)
        {
            ptr[i] = static_cast<T>( *(pImg+i) );
        }

        return OP_STATUS::SUCCESS;
    }

    /**
     * add data from std::vector
     */
    template<typename T>
    OP_STATUS Tensor<T>::addData(std::vector<T> data) 
    {


        if (data.size() != wholeSize_)
        {
            return OP_STATUS::UNMATCHED_SIZE;
        }

        // Get current write address
        T* ptr = pData_;

        for (int i = 0; i < wholeSize_; ++i)
        {
            ptr[i] = static_cast<T>( data.at(i) );
        }

        return OP_STATUS::SUCCESS;
    }

    template<typename T>
    OP_STATUS Tensor<T>::addOneSample(char const *filename)
    {



        // Load image from file
        int w, h, c;
        unsigned char *pImg = stbi_load(filename, &w, &h, &c, 0);

        int image_size =  w*h*c;
        if (w != width_ || h != height_ || c != channel_)
        {
            return OP_STATUS::UNMATCHED_SIZE;
        }

        if (wrCount_ + image_size > wholeSize_)
        {
            return OP_STATUS::OVER_MAX_SIZE;
        }

        // fprintf(stderr, "w: %d, h: %d, c: %d\n", w, h, c);

        T* ptr = pData_ + wrCount_;
        for (int i = 0; i < image_size; ++i)
        {
            *(ptr+i) = static_cast<T>( *(pImg+i) );
        }
        wrCount_ += image_size;
        return OP_STATUS::SUCCESS;
    }

    template<typename T>
    OP_STATUS Tensor<T>::addOneSample(const T *pSample, int size)
    {
        // the size of one sample is channel * height * width = size3D
        if (size != size3D_)
        {
            return OP_STATUS::UNMATCHED_SIZE;
        }

        if (wrCount_ + size > wholeSize_)
        {
            return OP_STATUS::OVER_MAX_SIZE;
        }


        T* ptr = pData_ + wrCount_;
        for (int i = 0; i < size; ++i)
        {
            *(ptr+i) = static_cast<T>( *(pSample+i) );
        }
        wrCount_ += size;
        return OP_STATUS::SUCCESS;
    }

     template<typename T>
    OP_STATUS Tensor<T>::addOneSample(const std::vector<T> vSample)
    {
        // the size of one sample is channel * height * width = size3D
        if (vSample.size() != size3D_)
        {
            return OP_STATUS::UNMATCHED_SIZE;
        }

        if (wrCount_ + vSample.size() > wholeSize_)
        {
            return OP_STATUS::OVER_MAX_SIZE;
        }

        T* ptr = pData_ + wrCount_;
        for (int i = 0; i < vSample.size(); ++i)
        {
            *(ptr+i) = static_cast<T>( vSample[i] );
        }

        wrCount_ += vSample.size();
        return OP_STATUS::SUCCESS;
    }

    // Reset data in tensor to zero
    template<typename T>
    void Tensor<T>::resetData() {
        // std::memset(pData_, 0, wholeSize_ * sizeof(float));
        std::fill_n(pData_, wholeSize_, 0);
    }

    /********************************
    ** Getter
    ********************************/
    template<typename T>
    T* Tensor<T>::getCPUData()       { return pData_; }
    template<typename T>
    int    Tensor<T>::getNumOfData() { return num_; }
    template<typename T>
    int    Tensor<T>::getWidth()     { return width_; }
    template<typename T>
    int    Tensor<T>::getHeight()    { return height_; }
    template<typename T>
    int    Tensor<T>::getChannel()   { return channel_; }
    template<typename T>
    int    Tensor<T>::getSize2D()    { return size2D_; }
    template<typename T>
    int    Tensor<T>::getSize3D()    { return size3D_; }
    template<typename T>
    int    Tensor<T>::getWholeSize() { return wholeSize_; }

    template<typename T>
    Shape Tensor<T>::getShape() 
    {
        Shape shape{num_, channel_, height_, width_};
        return shape;
    }


    // Serialize / Deserialize function
    template<typename T>
    void Tensor<T>::serialize(std::fstream& file, bool bWriteInfo)
    {
        file.write(reinterpret_cast<const char*>(pData_), sizeof(T)*wholeSize_);
    }

    template<typename T>
    void Tensor<T>::deserialize(std::fstream& file, bool bReadInfo)
    {
        file.read(reinterpret_cast<char*>(pData_), sizeof(T)*wholeSize_);
    }


    // Explicitly instantiate the template, and its member definitions
    template class Tensor<float>;

} // namespace mkt

