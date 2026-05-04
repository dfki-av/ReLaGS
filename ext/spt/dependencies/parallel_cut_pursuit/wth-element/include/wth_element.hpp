/*=============================================================================
 * W-th element: weighted n-th element
 *
 * The 'weighted rank interval' of an element e can be defined as the interval
 *
 *      ]wsum(e), wsum(e) + weight(e)]
 *
 * where wsum(e) is the cumulative sum of the weights of all elements comparing
 * lower to e, and weight(e) is the weight associated to e.
 * The w-th element is the element whose weighted rank interval contains w;
 * note that if all weights are equal to unity, the w-th element with w = n
 * reduces to the n-th element (starting count at 0).
 *
 * Based on quickselect algorithm
 *
 * Hugo Raguet 2018
 *===========================================================================*/
#pragma once
#include <limits>

/**  macros common to all versions  **/
#define SWAP(i, j) auto tmp = ARRAY[i]; ARRAY[i] = ARRAY[j]; ARRAY[j] = tmp
#define VALUE(i) values[IDX(i)]
#define COMP(i, j) (VALUE(i) < VALUE(j))

/**  macros for non-weighted versions  **/
#define WEIGHT(i) 1
#define INCR incr++
#define WRK_INCR incr

/* non-weighted, no index:
 * 'values' is an array of values to be compared;
 * the rank 'wrk' is actually the rank (starting at 0 for the lowest) of the
 * element in the array, so expected to be an index between 0 and size - 1;
 * returns the value of the wrk-th element;
 * 'values' is reordered such that the wrk-th element is at index wrk */
#define ARRAY values
#define IDX(i) i
template <typename value_t, typename index_t, typename rank_t>
value_t nth_element(value_t* values, index_t size, rank_t wrk)
#include "../src/wth_element_generic.cpp"

/* non-weighted, with indices:
 * like the non-weighted version above, but the values are read-only, indexed
 * by the indices in the dedicated array;
 * returns the value of the wrk-th element;
 * the index of the wrk-th element is at index wrk within 'indices' */
#undef ARRAY
#undef IDX
#define ARRAY indices
#define IDX(i) indices[i]
template <typename value_t, typename index_t, typename rank_t>
value_t nth_element_idx(index_t* indices, const value_t* values, index_t size,
    rank_t wrk)
#include "../src/wth_element_generic.cpp"

/* weighted version, requires indices
 * 'values' is an array of values to be compared;
 * 'weights' is an array of weighting the values;
 * the rank 'wrk' is a weighted rank, expected to range from zero up to the
 * total sum of all weights;
 * returns the value of the element whose weighted rank interval contains 'wrk'
 * 'values' and 'weights' are read-only, indexed by the indices in the
 * dedicated array; 'indices' is reordered such that all indices before the
 * index of the wrk-th element refers to lower values, and all indices after
 * the wrk-th refers to greater values */
#undef WEIGHT
#undef INCR
#undef WRK_INCR
#define WEIGHT(i) weights[IDX(i)]
#define INCR wrk_incr += WEIGHT(incr); incr++
#define WRK_INCR wrk_incr
template <typename value_t, typename index_t, typename rank_t,
    typename weight_t>
value_t wth_element(index_t* indices, const value_t* values, index_t size,
    rank_t wrk, const weight_t* weights)
#include "../src/wth_element_generic.cpp"

/* functions have been defined, all these macros are useless now */
#undef SWAP
#undef VALUE
#undef COMP
#undef WEIGHT
#undef INCR
#undef WRK_INCR
#undef ARRAY
#undef IDX
