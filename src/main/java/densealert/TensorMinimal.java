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

/**
 * A data structure for storing tensor (minimal feature)
 * @author kijungs
 */
class TensorMinimal {

    public int order;
    public int[][][][] modeToAttValToEntries;
    public int[][] modeToAttValToDegree;
    public int[][] modeToAttValToCardinality;

    public TensorMinimal(int order, int[] modeToIndicesNum) {
        this.order = order;
        modeToAttValToEntries = new int[order][][][];
        modeToAttValToDegree = new int[order][];
        modeToAttValToCardinality = new int[order][];
        for(int mode = 0; mode < order; mode++) {
            modeToAttValToDegree[mode] = new int[modeToIndicesNum[mode]];
            modeToAttValToCardinality[mode] = new int[modeToIndicesNum[mode]];
            modeToAttValToEntries[mode] = new int[modeToIndicesNum[mode]][][];
        }
    }

    public void resize(int mode, int newLength) {

        // degree
        modeToAttValToDegree[mode] = new int[newLength];

        //cardinality
        modeToAttValToCardinality[mode] = new int[newLength];

        // entries
        {
            final int[][][] oldArray = modeToAttValToEntries[mode];
            int oldLength = oldArray.length;
            final int[][][] newArray = new int[newLength][][];
            modeToAttValToEntries[mode] = newArray;
            for (int i = 0; i < oldLength; i++) {
                newArray[i] = oldArray[i];
            }
        }
    }

    public void resize(int mode, int attVal, int newLength) {
        modeToAttValToEntries[mode][attVal] = new int[newLength][];
    }

    /**
     * insert without considering resizing
     * @param entry
     * @param insertFlag whether each attribute should be actually inserted
     */
    public void insert(int[] entry, boolean[] insertFlag) {
        for(int mode=0; mode < order; mode++) {
            if(insertFlag[mode]) {
                int attVal = entry[mode];
                int[][][] attValToEntries = modeToAttValToEntries[mode];
                int[] attValToDegree = modeToAttValToDegree[mode];
                int[] attValToCardinality = modeToAttValToCardinality[mode];
                int cardinality = attValToCardinality[attVal];
                if(cardinality==0) { // new entry
                    if(attValToEntries[attVal]==null) {
                        attValToEntries[attVal] = new int[4][];
                    }
                }
                attValToEntries[attVal][cardinality] = entry;
                attValToDegree[attVal] += entry[order];
                attValToCardinality[attVal] += 1;
            }
        }
    }

    /**
     * do not add an entry, but increase its degree as it is added
     * @param entry
     * @param mode
     */
    public void addDegree(int[] entry, int mode) {
        int attVal = entry[mode];
        int increment = entry[order];

        int[] attValToDegree = modeToAttValToDegree[mode];
        attValToDegree[attVal] += increment;
    }

    /**
     * remove the given attribute from the given mode
     * @param mode
     * @param attVal
     */
    public void deleteAttVal(int mode, int attVal) {
        modeToAttValToDegree[mode][attVal] = 0;
        modeToAttValToCardinality[mode][attVal] = 0;
    }
}
