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
import java.util.List;
import java.util.Map;

/**
 * DenseStream
 *
 * @author kijungs
 */
public class DenseStream {

    private IndexMatching indexMatching;
    private TensorFull tensor;
    private Core core;
    private int order;
    private int arrayLength;
    private Map<Integer, int[]> blockIndices = new HashMap<Integer, int[]>();

    /**
     *
     * @param order order of the input tensor
     */
    public DenseStream(int order){
        this.order = order;
        this.indexMatching = new IndexMatching(order);
        this.tensor = new TensorFull(order, indexMatching.modeToIndicesNum);
        this.core = new Core(tensor);
        this.arrayLength = order * 2 + 2;
    }

    /**
     * processing insertion/increment
     * @param insertedEntry (i_{1}, i_{2}, ..., i_{N}, Delta)
     */
    public void insert(int[] insertedEntry) {
        int[] entry = new int[arrayLength];
        for(int dim = 0; dim < order; dim++) {
            entry[dim] = insertedEntry[dim];
        }
        entry[order] = insertedEntry[order];
        entry = indexMatching.changeToIndex(entry);
        core.insert(entry);
    }

    /**
     * processing deletion/decrement
     * @param deletedEntry (i_{1}, i_{2}, ..., i_{N}, Delta)
     */
    public void delete(int[] deletedEntry) {
        int[] entry = new int[arrayLength];
        for(int dim = 0; dim < order; dim++) {
            entry[dim] = deletedEntry[dim];
        }
        entry[order] = deletedEntry[order];
        entry = indexMatching.changeToIndex(entry);
        core.delete(entry);
    }

    /**
     * get density of the maintained block
     * @return
     */
    public double getDensity() {
        return core.getDensity();
    }

    /**
     * get mode and indices of the input tensor composing the densest block
     * @return mode to list of indices forming a dense block
     */
    public Map<Integer, int[]> getBlockIndices() {

        if(core.isBlockChanged()) {
            List<Integer>[] maintainedBlock = core.getDenseBlockAttVals();
            if(maintainedBlock == null) {
                int[][] modeToAttValToCardinality = tensor.modeToAttValToCardinality;
                int[][] modeToIndexToId = indexMatching.modeToIndexToId;

                for(int mode = 0; mode < order; mode++) {
                    int[] indexToCardinality = modeToAttValToCardinality[mode];
                    int[] indexToId = modeToIndexToId[mode];

                    int count = 0;
                    int length = indexToCardinality.length;
                    for(int index = 0; index<length; index++) {
                        if(indexToCardinality[index] > 0 ){
                            count++;
                        }
                    }

                    int[] ids = new int[count];
                    int loc = 0;
                    for(int index = 0; index<length; index++) {
                        if(indexToCardinality[index] > 0 ){
                            ids[loc++] = indexToId[index];
                        }
                    }
                    blockIndices.put(mode, ids);
                }
            }
            else {
                int[][] modeToIndexToId = indexMatching.modeToIndexToId;
                for(int mode = 0; mode < order; mode++) {
                    int[] indexToId = modeToIndexToId[mode];
                    int[] ids = new int[maintainedBlock[mode].size()];
                    int loc = 0;
                    for(int index : maintainedBlock[mode]) {
                        ids[loc++] = indexToId[index];
                    }
                    blockIndices.put(mode, ids);
                }

            }
            core.setBlockChanged(false);
        }
        return blockIndices;
    }

}
