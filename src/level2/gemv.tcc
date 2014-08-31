#ifndef ULMBLAS_SRC_LEVEL2_GEMV_TCC
#define ULMBLAS_SRC_LEVEL2_GEMV_TCC 1

#include <src/auxiliary/memorypool.h>
#include <src/config/blocksize.h>
#include <src/level1/axpy.h>
#include <src/level1/copy.h>
#include <src/level1/dot.h>
#include <src/level1/scal.h>
#include <src/level1extensions/axpy2v.h>
#include <src/level1extensions/axpyf.h>
#include <src/level1extensions/dotxf.h>
#include <src/level1extensions/gecopy.h>
#include <src/level2/gemv.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
void
gemv(IndexType    m,
     IndexType    n,
     const Alpha  &alpha,
     const TA     *A,
     IndexType    incRowA,
     IndexType    incColA,
     const TX     *x,
     IndexType    incX,
     const Beta   &beta,
     TY           *y,
     IndexType    incY)
{
    typedef decltype(Alpha(0)*TA(0)*TX(0)+Beta(0)*TY(0))  T;

    const IndexType    UnitStride(1);
    static const bool  homogeneousTypes = std::is_same<T,Alpha>::value
                                       && std::is_same<T,TA>::value
                                       && std::is_same<T,TX>::value
                                       && std::is_same<T,TY>::value;

    if (m<=0 || n<=0 || (alpha==Alpha(0) && beta==Beta(1))) {
        return;
    }

    scal(m, beta, y, incY);

    if (alpha==Alpha(0)) {
        return;
    }

//
//  If all operands have the same element type and matrix A is col major we use
//  fused axpy operations.
//
    if (homogeneousTypes && incRowA==UnitStride) {
        const IndexType bf = axpyf_fusefactor<T>();
        const IndexType nb = (n/bf)*bf;

        for (IndexType j=0; j<nb; j+=bf) {
            axpyf(m, alpha, &x[j*incX], incX,
                  &A[j*incColA], UnitStride, incColA,
                  y, incY);
        }
        for (IndexType j=nb; j<n; ++j) {
            axpy(m, alpha*x[j*incX], &A[j*incColA], UnitStride, y, incY);
        }
//
//  If all operands have the same element type and matrix A is row major we use
//  fused dot operations.
//
    } else if (homogeneousTypes && incColA==UnitStride) {
        const IndexType bf = dotuxf_fusefactor<T>();
        const IndexType mb = (m/bf)*bf;

        TY tmp[bf];

        for (IndexType i=0; i<mb; i+=bf) {
            dotuxf(n, &A[i*incRowA], incRowA, UnitStride,
                      x, incX,
                      tmp, UnitStride);
            for (IndexType l=0; l<bf; ++l) {
                y[(i+l)*incY] += alpha*tmp[l];
            }
        }
        for (IndexType i=mb; i<m; ++i) {
            dotu(n, &A[i*incRowA], UnitStride, x, incX, tmp[0]);
            y[i*incY] += alpha*tmp[0];
        }
    } else {
//
//  Otherwise we pack operands.
//
        /*
        // Simple reference implementation
        for (IndexType j=0; j<n; ++j) {
            for (IndexType i=0; i<m; ++i) {
                y[i*incY] += A[i*incRowA+j*incColA]*x[j*incX];
            }
        }
        */

        static MemoryPool<T> memoryPool;
        //const bool packA    = !((incRowA==UnitStride || incColA==UnitStride) &&
        //                      std::is_same<T,TA>::value);
        //const bool packX    = !(incX==UnitStride && std::is_same<T,TX>::value);
        //const bool packY    = !(incY==UnitStride && std::is_same<T,TY>::value);

        const bool packA    = true;
        const bool packX    = true;
        const bool packY    = true;

        //printf("packed: packX=%d, packY=%d, packA=%d\n", packX, packY, packA);

        const IndexType MC  = BlockSize<T>::MC;
        const IndexType NC  = BlockSize<T>::NC;

        const T &_alpha     = alpha;
        T *buffer_A         = packA ? memoryPool.allocate(MC*NC) : 0;
        T *buffer_x         = packX ? memoryPool.allocate(NC)    : 0;
        T *buffer_y         = packY ? memoryPool.allocate(MC)    : 0;

        const T *_A         = packA ? buffer_A : 0;
        const T *_x         = packX ? buffer_x : 0;

        const IndexType mb  = (m+MC-1) / MC;
        const IndexType nb  = (n+NC-1) / NC;

        const IndexType _mc = m % MC;
        const IndexType _nc = n % NC;

        for (IndexType j=0; j<nb; ++j) {
            IndexType nc = (j!=nb-1 || _nc==0) ? NC : _nc;

            if (packX) {
                copy(nc, &x[j*NC*incX], incX, buffer_x, UnitStride);
            } else {
                _x = &x[j*NC];
            }

            for (IndexType i=0; i<mb; ++i) {
                IndexType mc = (i!=mb-1 || _mc==0) ? MC : _mc;
                IndexType incRow_A, incCol_A;

                if (packA) {
                    incRow_A = UnitStride;
                    incCol_A = mc;
                    gecopy(mc, nc,
                           &A[i*MC*incRowA+j*NC*incColA], incRowA, incColA,
                           buffer_A, incRow_A, incCol_A);
                } else {
                    incRow_A = incRowA;
                    incCol_A = incColA;
                    _A = &A[i*MC*incRowA+j*NC*incColA];
                }

                if (packY) {
                    gemv(mc, nc, _alpha,
                         _A, incRow_A, incCol_A,
                         _x, UnitStride,
                         T(0),
                         buffer_y, UnitStride);

                    axpy(mc, T(1), buffer_y, UnitStride, &y[i*MC*incY], incY);
                } else {
                    gemv(mc, nc, _alpha,
                         _A, incRow_A, incCol_A,
                         _x, UnitStride,
                         T(1),
                         &y[i*MC*incY], UnitStride);
                }
            }
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL2_GEMV_TCC
