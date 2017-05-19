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
 * Hash indexed min heap
 * @author kijungs
 */
class HashIndexedMinHeap {

    /**
     * heap: array of keys
     */
    private int[] array;

    /**
     * Number of objects in the heap
     */
    private int size;

    /**
     * Maximum number of objects in the heap
     */
    private int capacity;

    /**
     * index -> position
     */
    private int[] positions;

    /**
     * index -> value
     */
    private int[] values;

    /**
     * Position indicates that keys do not exist
     */
    private final int missingPosition = -1;

    public HashIndexedMinHeap(int capacity){
        this.capacity = capacity;
        this.array = new int[capacity];
        this.values = new int[capacity];
        this.positions = new int[capacity];
        this.size = 0;
        for(int i = 0; i < capacity; i++) {
            this.positions[i] = missingPosition;
        }
    }

    public int size(){
        return size;
    }

    public boolean containsKey(int key){
        return (positions[key] == missingPosition) ? false : true;
    }

    public int[] peek(){
        if(size == 0){
            return null;
        }
        return new int[]{array[0], values[array[0]]};
    }

    public int[] poll(){

        if(size == 0){
            return null;
        }

        int[] top = this.peek();
        positions[top[0]] = missingPosition;

        if(size != 1){
            int last = array[size-1];
            array[0] = last;
            positions[last] = 0;

            size--;
            this.minHeapfy(0);
        }
        else{
            size--;
        }
        array[size] = 0;

        return top;
    }

    public boolean insert(int key, int value){

        if(size >= capacity)
            return false;

        int pos = size;
        size++;
        array[pos] = key;
        positions[key] = pos;
        values[key] = value;
        this.refreshPriority(key, value);
        return true;
    }

    public int getPriority(int key){
        return values[key];
    }

    public void refreshPriority(int key, int value){

        values[key] = value;
        int pos = positions[key];
        boolean shiftedDown = this.minHeapfy(pos);

        if(!shiftedDown){
            if(pos > 0){
                int cur = key;
                int parentPos = ((pos + 1) / 2) - 1;
                int pel = array[parentPos];
                while(pos > 0 && values[pel] > values[cur]){
                    array[parentPos] = cur;
                    positions[cur] = parentPos;
                    array[pos] = pel;
                    positions[pel] = pos;
                    pos = parentPos;
                    parentPos = ((pos + 1) / 2) - 1;
                    if(pos > 0){
                        pel = array[parentPos];
                    }
                }

            }
        }
    }

    private boolean minHeapfy(int pos){

        int posLeft = (2*(pos+1))-1;
        int posRight = (2*(pos+1));

        int keyCur = array[pos];

        int smallest = pos;
        int nsmallest = keyCur;

        if(posLeft < size){
            int keyLeft = array[posLeft];
            if(values[keyLeft] < values[keyCur]){
                smallest = posLeft;
                nsmallest = keyLeft;
            }

        }

        if(posRight < size){
            int keyRight = array[posRight];
            if(values[keyRight] < values[nsmallest]){
                smallest = posRight;
                nsmallest = keyRight;
            }
        }

        if(smallest != pos){

            array[pos] = nsmallest;
            positions[nsmallest] = pos;

            array[smallest] = keyCur;
            positions[keyCur] = smallest;

            this.minHeapfy(smallest);
            return true;
        }

        return false;
    }
}