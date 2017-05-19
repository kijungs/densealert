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
 * A table for storing \pi, d_{\pi}, c_{\pi}
 * @author kijungs
 */
class Table {

    public TableCol head = null;
    public TableCol tail = null;

    // mode, attribute value -> corresponding column in a table
    public TableCol[][] modeToAttValToCol;

    // mode, attribute value -> core number of the attribute
    public int[][] modeToAttValToCoreNumber;

    public Table(int order, int[] modeToIndexNum) {
        modeToAttValToCol = new TableCol[order][];
        modeToAttValToCoreNumber = new int[order][];
        for(int dim = 0; dim < order; dim++) {
            modeToAttValToCol[dim] = new TableCol[modeToIndexNum[dim]];
            modeToAttValToCoreNumber[dim] = new int[modeToIndexNum[dim]];
        }
    }

    /**
     * add the given column to head
     * @param col
     */
    public void addToHead(TableCol col) {
        if(head == null) {
            head = col;
            tail = col;
        }
        else {
            head.prev = col;
            col.next = head;
            head = col;
        }
        modeToAttValToCol[col.mode][col.attVal] = col;
        modeToAttValToCoreNumber[col.mode][col.attVal] = col.coreNumber;
    }

    /**
     * remove the given column
     * @param col
     */
    public void delete(TableCol col) {
        modeToAttValToCol[col.mode][col.attVal] = null;
        modeToAttValToCoreNumber[col.mode][col.attVal] = 0;
        if (col==head) {
            head = col.next;
            if(col.next != null) {
                col.next.prev = col.prev;
            }
        }
        else {
            col.prev.next = col.next;
            if(col.next != null) {
                col.next.prev = col.prev;
            }
        }

        if(col==tail) {
            tail = col.prev;
        }

    }

    /**
     * add the given column to tail
     * @param col
     */
    public void addToTail(TableCol col) {
        if(tail == null) {
            head = col;
            tail = col;
        }
        else {
            tail.next = col;
            col.prev = tail;
            tail = col;
        }
        modeToAttValToCol[col.mode][col.attVal] = col;
        modeToAttValToCoreNumber[col.mode][col.attVal] = col.coreNumber;
    }

    /**
     * resize the table
     * @param mode
     * @param newSize
     */
    public void resize(int mode, int newSize) {
        {
            final TableCol[] oldArray = modeToAttValToCol[mode];
            modeToAttValToCol[mode] = new TableCol[newSize];
            final TableCol[] newArray = modeToAttValToCol[mode];
            final int oldSize = oldArray.length;
            for(int i=0; i<oldSize; i++) {
                newArray[i] = oldArray[i];
            }
        }
        {
            final int[] oldArray = modeToAttValToCoreNumber[mode];
            modeToAttValToCoreNumber[mode] = new int[newSize];
            final int[] newArray = modeToAttValToCoreNumber[mode];
            final int oldSize = oldArray.length;
            for(int i=0; i<oldSize; i++) {
                newArray[i] = oldArray[i];
            }
        }

    }

}
