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

import java.util.*;

/**
 * DenseAlert Simulation
 *
 * @author kijungs
 */
public class DenseAlert {

    private IndexMatching indexMatching;
    private TensorFull tensor;
    private Core core;
    private int order;
    private int arrayLength;
    private int window;
    private Map<Integer, int[]> blockIndices = new HashMap<Integer, int[]>();
    private Queue<Pair<Long, int[]>> deleteQueue = new LinkedList();

    /**
     * @param order order of the input tensor
     * @param window size of window (in seconds)
     */
    public DenseAlert(int order, int window){
        this.order = order;
        this.window = window;
        this.indexMatching = new IndexMatching(order);
        this.tensor = new TensorFull(order, indexMatching.modeToIndicesNum);
        this.core = new Core(tensor);
        this.arrayLength = order * 2 + 2;
    }

    /**
     * processing insertion/increment
     * @param insertedEntry (i_{1}, i_{2}, ..., i_{N}, Delta)
     * @param timestamp
     */
    public void insert(int[] insertedEntry, long timestamp) {

        while(!deleteQueue.isEmpty() && deleteQueue.peek().getKey() < timestamp) {
            Pair<Long, int[]> pair = deleteQueue.poll();
            int[] entryToDelete = pair.getValue();
            core.delete(entryToDelete);
        }

        int[] entry = new int[arrayLength];
        for(int dim = 0; dim < order; dim++) {
            entry[dim] = insertedEntry[dim];
        }

        entry[order] = insertedEntry[order];
        entry = indexMatching.changeToIndex(entry);
        core.insert(entry);
        deleteQueue.add(new Pair<Long, int[]>(timestamp + window, entry.clone()));
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
        return null;
    }


}
