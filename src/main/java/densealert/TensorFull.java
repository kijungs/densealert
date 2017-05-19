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

import java.util.Arrays;

/**
 * A data structure for storing tensor (full feature)
 * @author kijungs
 */
class TensorFull {

    public int order;
    public long mass;
    public long omega;
    public int cardinality;
    public int[][][][] modeToAttValToEntries;
    public int[][] modeToAttValToDegree;
    public int[][] modeToAttValToCardinality;
    private int locateBase = order+2;

    public TensorFull(int order, int[] modeToIndicesNum) {
        this.order = order;
        locateBase = order+2;
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
        modeToAttValToDegree[mode] = Arrays.copyOf(modeToAttValToDegree[mode], newLength);

        // cardinality
        modeToAttValToCardinality[mode] = Arrays.copyOf(modeToAttValToCardinality[mode], newLength);

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

    /**
     * insert the given entry or increment the value if exist
     * @param entry
     * @return
     */
    public int[] insert(int[] entry) {

        for(int mode = 0; mode < order; mode++) {
            int attVal = entry[mode];
            if(attVal == modeToAttValToDegree[mode].length) {
                resize(mode, modeToAttValToDegree[mode].length * 2);
            }
        }

        int[] modeToNewLength = new int[order];

        //check whether the same entry exists
        int minMode = 0;
        int minEntryNum = Integer.MAX_VALUE;
        for(int mode = 0; mode < order; mode++) {
            int entryNum = modeToAttValToCardinality[mode][entry[mode]];
            if(entryNum < minEntryNum) {
                minMode = mode;
                minEntryNum = entryNum;
            }
        }
        int[][] entries = modeToAttValToEntries[minMode][entry[minMode]];
        out:for(int i=minEntryNum-1; i>=0; i--) { //from last one
            int[] existingEntry = entries[i];
            for(int mode = 0; mode < order; mode++) {
                if(existingEntry[mode] != entry[mode])
                    continue out;
            }
            //exists
            existingEntry[order] += entry[order];
            mass += entry[order];
            for(int mode = 0; mode < order; mode++) {
                modeToAttValToDegree[mode][entry[mode]] += entry[order];
            }
            return modeToNewLength;
        }

        for(int mode=0; mode < order; mode++) {
            int attVal = entry[mode];
            int[][][] attValToEntries = modeToAttValToEntries[mode];
            int[] attValToDegree = modeToAttValToDegree[mode];
            int[] attValToCardinality = modeToAttValToCardinality[mode];
            int cardinality = attValToCardinality[attVal];
            if(cardinality==0) {
                if(attValToEntries[attVal]==null) {
                    attValToEntries[attVal] = new int[4][];
                }
                this.cardinality += 1;
            }
            else if(cardinality == attValToEntries[attVal].length) {
                //resize
                int oldLength = attValToEntries[attVal].length;
                int newLength = oldLength * 2;
                int[][] oldArray = attValToEntries[attVal];
                attValToEntries[attVal] = new int[newLength][];
                int[][] newArray = attValToEntries[attVal];
                for(int i=0; i < oldLength; i++) {
                    newArray[i] = oldArray[i];
                }
                modeToNewLength[mode] = newLength;
            }

            entry[locateBase + mode] = cardinality;
            attValToEntries[attVal][cardinality] = entry;
            attValToDegree[attVal] += entry[order];
            attValToCardinality[attVal] += 1;

        }
        mass += entry[order];
        omega += 1;

        return modeToNewLength;
    }

    /**
     * delete the given entry or decrement the value if exist
     * @param entry
     * @return null if non corresponding entry is found;
     */
    public int[] delete(int[] entry) {

        int[] modeToNewLength = new int[order];

        //check whether the same entry exists
        int minMode = 0;
        int minEntryNum = Integer.MAX_VALUE;
        for(int mode = 0; mode < order; mode++) {
            int entryNum = modeToAttValToCardinality[mode][entry[mode]];
            if(entryNum < minEntryNum) {
                minMode = mode;
                minEntryNum = entryNum;
            }
        }
        int minAttVal = entry[minMode];
        int[][] entries = modeToAttValToEntries[minMode][minAttVal];

        int[] entryToRemove = null;
        out:for(int minModeIndex=0; minModeIndex<minEntryNum; minModeIndex++) {
            int[] existingEntry = entries[minModeIndex];
            for(int mode = 0; mode < order; mode++) {
                if(existingEntry[mode] != entry[mode])
                    continue out;
            }
            entryToRemove = existingEntry; //found
            break out;
        }

        if(entryToRemove == null) { // no entry is found
            return null;
        }
        else if(entryToRemove[order] > entry[order]) { // only change value
            entryToRemove[order] -= entry[order];
            mass -= entry[order];
            for(int mode = 0; mode < order; mode++) {
                modeToAttValToDegree[mode][entry[mode]] -= entry[order];
            }
            return modeToNewLength;
        }
        else { //should remove entry
            int massToRemove = entryToRemove[order];
            for(int mode = 0; mode < order; mode++) {

                int attVal = entry[mode];
                int entryNum = modeToAttValToCardinality[mode][attVal]--;
                modeToAttValToDegree[mode][attVal] -= massToRemove;
                entries = modeToAttValToEntries[mode][attVal];

                int modeIndex = entryToRemove[locateBase + mode];

                // relocate
                int[] entryToMove = modeToAttValToEntries[mode][attVal][entryNum - 1];
                entryToMove[locateBase + mode] = modeIndex;
                modeToAttValToEntries[mode][attVal][modeIndex] = entryToMove;

                //if this attribute value should be removed
                if(entryNum == 1) {
                    modeToNewLength[mode] = -1; //remove
                    modeToAttValToEntries[mode][attVal] = null;
                    this.cardinality--;
                }
                else if (entryNum >= 4 && entryNum < entries.length / 4) { //should be resized
                    int[][][] attValToEntries = modeToAttValToEntries[mode];
                    int oldLength = entries.length;
                    int newLength = oldLength / 2;
                    int[][] oldArray = entries;
                    int[][] newArray = new int[newLength][];
                    attValToEntries[attVal] = newArray;
                    for (int i = 0; i < entryNum; i++) {
                        newArray[i] = oldArray[i];
                    }
                    modeToNewLength[mode] = newLength;
                }
            }
            mass -= massToRemove;
            this.omega--;
            return modeToNewLength;
        }
    }

}
