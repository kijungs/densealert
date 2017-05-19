/* =================================================================================
 *
 * DenseAlert: Incremental Dense-Block Detection in Tensor Streams
 * Authors: Kijung Shin, Bryan Hooi, Jisu Kim, and Christos Faloutsos
 *
 * Version: 1.0
 * Date: Oct 24, 2016
 * Main Contact: Kijung Shin (kijungs@cs.cmu.edu)
 *
 * This software is free of charge under research purposes.
 * For commercial purposes, please contact the author.
 *
 * =================================================================================
 */

package densealert;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;

/**
 * a module for matching real attribute value to an index
 * @author kijungs
 */
public class IndexMatching {

    public int order;

    // dim to id to index (0 ~ max node id)
    public HashMap<Integer, Integer>[] modeToIdToIndex;

    public Queue<Integer>[] modeToRemainedIndex;

    // dim to index (0 ~ max node id) to index
    public int[][] modeToIndexToId;

    // maximum node number
    public int[] modeToIndicesNum;

    public final int DEFAULT_SIZE = 1000;

    public IndexMatching(int order) {
        this.order = order;
        modeToIndicesNum = new int[order];
        modeToIdToIndex = new HashMap[order];
        modeToRemainedIndex = new Queue[order];
        modeToIndexToId = new int[order][];

        for(int dim = 0; dim < order; dim++) {
            modeToIndicesNum[dim] = DEFAULT_SIZE;
            modeToIdToIndex[dim] = new HashMap(DEFAULT_SIZE);
            modeToRemainedIndex[dim] = new LinkedList();
            modeToIndexToId = new int[order][DEFAULT_SIZE];
        }
    }

    /**
     * change the ids in an entry to indices
     * @param entry
     * @return
     */
    public int[] changeToIndex(int[] entry) {
        for(int dim = 0; dim < order; dim++) {
            int id = entry[dim];
            int index = 0;
            if(modeToIdToIndex[dim].containsKey(id)) {
                index = entry[dim] = modeToIdToIndex[dim].get(id);
            }
            else {
                index = modeToRemainedIndex[dim].isEmpty() ? modeToIdToIndex[dim].size() : modeToRemainedIndex[dim].poll();
                modeToIdToIndex[dim].put(id, index);
                modeToIndexToId[dim][index] = id;
            }

            //increase size
            if(index == modeToIndicesNum[dim] - 1) {
                int oldLength = modeToIndicesNum[dim];
                int newLength = oldLength * 2;
                modeToIndicesNum[dim] = newLength;
                int[] oldArray = modeToIndexToId[dim];
                modeToIndexToId[dim] = new int[newLength];
                int[] newArray = modeToIndexToId[dim];
                for(int i=0; i<oldLength; i++) {
                    newArray[i] = oldArray[i];
                }
            }

            entry[dim] = index;
        }
        entry[order] = entry[order];
        return entry;
    }

}
