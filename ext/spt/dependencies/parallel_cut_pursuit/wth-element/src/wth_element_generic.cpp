/* generic definition of wth_element and the likes
 * macros used:
 * SWAP(i, j) swaps the objects indexed by i and j;
 * WEIGHT(i) get the weight associated to index i;
 * COMP(i, j) compares indices i and j;
 * INCR increment the last index comparing lower than pivot;
 * WRK_INCR weighted rank of the last index comparing lower than pivot;
 */
{
    index_t first = 0, last = size - 1;
    rank_t wrk_first = 0, wrk_incr;
    /* avoid machine precision issue */
    wrk += wrk*std::numeric_limits<rank_t>::epsilon();
    while (last - first > 1){
        wrk_incr = wrk_first;
    
        /* select pivot with crude median search over three elements, and put
         * it at the begining of the current range; note that at least one
         * other index in the range compares greater or equal to the pivot */
        index_t middle = first + (last - first + 1)/2;
        if (COMP(first, middle)){
            if (COMP(middle, last)){ SWAP(first, middle); }
            else if (COMP(first, last)){ SWAP(first, last); }
        }else{
            if (COMP(last, middle)){ SWAP(first, middle); }
            else if (COMP(last, first)){ SWAP(first, last); }
        }

        /* partition according to pivot and weights */
        const index_t pivot = first;
        index_t incr = first + 1, decr = last;
        wrk_incr += WEIGHT(pivot);
        while (true){
            while (COMP(incr, pivot)){ INCR; }
            while (COMP(pivot, decr)){ decr--; }

            if (decr <= incr){ break; }

            SWAP(incr, decr);
            INCR;
            decr--;
        }

        /* at this point, all indices before decr compare lower to the pivot, 
         * and all indices after decr compare greater to the pivot, and it has
         * weighted rank equal to wrk_incr - WEIGHT(pivot) */
        if (WRK_INCR <= wrk){
            first = incr;
            wrk_first = WRK_INCR;
        }else if (WRK_INCR - WEIGHT(pivot) <= wrk){
            SWAP(pivot, decr); // put it in the right place
            return VALUE(decr);
        }else{
            SWAP(pivot, decr); // let's get rid of the pivot
            last = decr - 1;
        }
    }
    
    /* at this point, last is equal to, or successor of, first, and
     * at least one them should reach weighted rank wrk */
    if (!COMP(first, last)){ SWAP(first, last); }
    return wrk_first + WEIGHT(first) > wrk ? VALUE(first) : VALUE(last);
}
