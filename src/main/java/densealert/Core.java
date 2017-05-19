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
 * Core Module For Dense-Block Detection
 * @author kijungs
 */
class Core {

    private TensorFull oriTensor; // input tensor H
    private TensorMinimal subTensor; // subtensor formed for reordering

    private Table table; // table for intermediate data:  \pi, d_{\pi}, c_{\pi}
    private int order; // order of the input tensor
    private double density; //density of the densest block
    private int maintainedAttNum; //the number of attribute values in the maintained block

    // mode and attribute value which maximizes the density when it is removed
    int maxMode = -1;
    int maxAttVal = -1;

    //current densest block
    private List<Integer>[] maintainedBlock;

    //whether maintained block is changed
    private boolean isBlockChanged = false; 

    // core number -> (first column with the core number, (remained mass, num of remained att values) in the first column)
    private final HashMap<Integer, Pair<TableCol, long[]>> coreNumberToFirstColAndMass = new HashMap();

    // the number of attribute vlues in each node
    private int[] modeToAttValNum;

    // min heap for each mode
    private HashIndexedMinHeap[] modeToMinHeap;

    // mode, attVal -> 0: after wide range, 1: before wide range 2: in wide range 3: in narrow range
    private byte[][] modeToAttValToStatus;

    // mode -> list of attribute values in the wide range
    private int[][] modeToAttValsInWideRange;
    
    // mode -> list of attribute values in the narrow range
    private int[][] modeToAttValsInNarrowRange;
    
    // mode, attVal -> whether the attVal is in the densest block
    private boolean[][] modeToAttValToInMaintained;

    // array index used to check whether this entry (hyperedge) is processed or not
    private int indexForProcessed;


    /**
     * return the attribute values composing the dense block maintained
     * @return
     */
    List<Integer>[] getDenseBlockAttVals() {
        return maintainedBlock;
    }


    boolean isBlockChanged() {
        return isBlockChanged;
    }

    void setBlockChanged(boolean blockChanged) {
        isBlockChanged = blockChanged;
    }
    
    /**
     * return the density of the maintained block
     * @return
     */
    double getDensity() {
        return density;
    }
    
    /**
     * From an initial oriTensor
     * @param tensor
     */
    Core(TensorFull tensor) {
        
        this.oriTensor = tensor;
        this.order = tensor.order;
        this.indexForProcessed = order + 1;
        this.modeToAttValNum = new int[order];
        this.modeToMinHeap = new HashIndexedMinHeap[order];
        this.modeToAttValToStatus = new byte[order][];
        this.modeToAttValToInMaintained = new boolean[order][];
        this.modeToAttValsInWideRange = new int[order][];
        this.modeToAttValsInNarrowRange = new int[order][];

        for(int dim = 0; dim < order; dim++) {
            modeToAttValNum[dim] = tensor.modeToAttValToDegree[dim].length;
            modeToMinHeap[dim] = new HashIndexedMinHeap(modeToAttValNum[dim]);
            modeToAttValToStatus[dim] = new byte[modeToAttValNum[dim]];
            modeToAttValToInMaintained[dim] = new boolean[modeToAttValNum[dim]];
            modeToAttValsInWideRange[dim] = new int[modeToAttValNum[dim]];
            modeToAttValsInNarrowRange[dim] = new int[modeToAttValNum[dim]];
        }

        this.subTensor = createTensorWithSameSize(tensor, modeToAttValNum);
        if(tensor.omega > 0) {
            batch();
        }
    }

    TensorFull getTensor() {
        return oriTensor;
    }

    /**
     * batch algorithm for computing the densest block from the current tensor
     */
    private void batch() {

        final int[][][][] modeToAttValToEntries = oriTensor.modeToAttValToEntries;
        final int[][] modeToAttValToDegree = oriTensor.modeToAttValToDegree;
        final int[][] modeToAttValToCardinality = oriTensor.modeToAttValToCardinality;

        //make all entries unprocessed
        final int[][][] attValToEntries = modeToAttValToEntries[0];
        final int[] attValToCardinality = modeToAttValToCardinality[0];
        for(int attVal = 0; attVal < attValToEntries.length; attVal++) {
            int entryNum = attValToCardinality[attVal];
            if(entryNum > 0 ) {
                for (int index = 0; index < entryNum; index++) {
                    attValToEntries[attVal][index][indexForProcessed] = 0;
                }
            }
        }

        int n = 0;
        for(int dim = 0; dim < order; dim++) {
            HashIndexedMinHeap minHeap = modeToMinHeap[dim];
            int[] attValToDegree = modeToAttValToDegree[dim];
            for(int attVal = 0; attVal < attValToDegree.length; attVal++) {
                if(attValToDegree[attVal] > 0) {
                    minHeap.insert(attVal, attValToDegree[attVal]);
                    n++;
                }
            }
        }

        long mass = oriTensor.mass;
        density = ((double)mass) / n;
        int coreNumber = -1;

        table = new Table(order, modeToAttValNum);
        coreNumberToFirstColAndMass.clear();

        final int[][] modeToAttValToCoreNumber = table.modeToAttValToCoreNumber;
        int remainedNum = n;
        while(remainedNum > 0) {

            int minHeapMass = Integer.MAX_VALUE;
            int minDim = 0;
            for(int dim = 0; dim < order; dim++) {
                int[] pair = modeToMinHeap[dim].peek();
                if(pair != null && pair[1] < minHeapMass) {
                    minDim = dim;
                    minHeapMass = pair[1];
                }
            }

            int dim = minDim;
            int[] pair = modeToMinHeap[dim].poll();
            int attVal = pair[0];
            int removeMass = pair[1];

            TableCol col = new TableCol(dim, attVal, removeMass, coreNumber); //, mass, density);
            table.addToTail(col);
            if(removeMass > coreNumber) {
                col.coreNumber = removeMass;
                modeToAttValToCoreNumber[dim][attVal] = removeMass;
                for(int key = coreNumber+1; key <= removeMass; key++) {
                    coreNumberToFirstColAndMass.put(key, new Pair(col, new long[]{mass, remainedNum}));
                }
                coreNumber = removeMass;
            }

            double averageMass = ((double) mass) / remainedNum;
            if(averageMass > density) {
                density = averageMass;
                maxMode = dim;
                maxAttVal = attVal;
            }
            mass -= removeMass;
            remainedNum--;

            int cardinality = modeToAttValToCardinality[dim][attVal];
            int[][] indexToEntry = modeToAttValToEntries[dim][attVal];
            for(int i=0; i<cardinality; i++) {
                int[] entry = indexToEntry[i];
                if(entry[indexForProcessed] == 0) {
                    for (int _dim = 0; _dim < order; _dim++) {
                        if (_dim != dim) {
                            int key = entry[_dim];
                            HashIndexedMinHeap minHeap = modeToMinHeap[_dim];
                            if (minHeap.containsKey(key)) {
                                minHeap.refreshPriority(key, minHeap.getPriority(key) - entry[order]);
                            }
                        }
                    }
                    entry[indexForProcessed] = 1;
                }
            }
        }

        isBlockChanged = true;
        if(maintainedBlock !=null) {
            for (int dim = 0; dim < order; dim++) {
                for (int attVal : maintainedBlock[dim]) {
                    modeToAttValToInMaintained[dim][attVal] = false;
                }
            }
        }

        if(maxMode != -1) {
            maintainedAttNum = 0;
            TableCol[][] modeToAttValToCol = table.modeToAttValToCol;
            maintainedBlock = new List[order];
            for(int dim = 0; dim < order; dim++) {
                maintainedBlock[dim] = new LinkedList();
            }
            TableCol col = modeToAttValToCol[maxMode][maxAttVal];
            while(col != null) {
                maintainedBlock[col.mode].add(col.attVal);
                modeToAttValToInMaintained[col.mode][col.attVal] = true;
                maintainedAttNum++;
                col = col.next;
            }
        }
        else {
            maintainedBlock = null;
            maintainedAttNum = -1;
        }
    }
    
    /**
     * insert a new entry (or increment the value if exists) to the input tensor, and updated the densest block
     * @param newEntry
     */
    void insert(int[] newEntry){

        newEntry = newEntry.clone();

        if(newEntry[order] == 0) { // ignore
            return;
        }

        //resize data structures if necessary
        for(int dim = 0; dim < order; dim++) {

            //increase size
            if(newEntry[dim] == modeToAttValNum[dim] - 1) {
                int oldLength = modeToAttValNum[dim];
                int newLength = oldLength * 2;
                modeToAttValNum[dim] = newLength;
                modeToMinHeap[dim] = new HashIndexedMinHeap(newLength);
                modeToAttValToStatus[dim] = new byte[newLength];
                modeToAttValToInMaintained[dim] = Arrays.copyOf(modeToAttValToInMaintained[dim], newLength); // should be preserved
                modeToAttValsInWideRange[dim] = new int[newLength];
                modeToAttValsInNarrowRange[dim] = new int[newLength];
                table.resize(dim, newLength);
                oriTensor.resize(dim, newLength);
                subTensor.resize(dim, newLength);
            }
        }

        if(oriTensor.omega == 0) {
            oriTensor.insert(newEntry);
            batch();
            return;
        }

        // density before insertion
        final double prevMaxDensity = density;

        // add the new entry in a tensor
        final int[] modeToNewLength = oriTensor.insert(newEntry);
        for(int dim = 0; dim < order; dim++) {
            if(modeToNewLength[dim] > 0) {
                subTensor.resize(dim, newEntry[dim], modeToNewLength[dim]);
            }
        }
        
        final int value = newEntry[order];

        final int[] minMaxC = findMinMaxCoreNumberForInsertion(newEntry);

        // minimum core number we should look at for reordering
        final int minCReorder = minMaxC[0];

        // maximum core number we should look at for reordering
        final int maxCReorder = minMaxC[1];

        // number of remained attribute values (not removed yet)
        long remainedNum = oriTensor.cardinality;

        // density of the entire block
        double currentDensity = ((double) oriTensor.mass) / remainedNum;
        if(currentDensity > density) {
            density = currentDensity;
            maxMode = -1;
            maxAttVal = -1;
        }
        
        // current mass
        long mass = oriTensor.mass;

        // minimum core number that we should inspect for finding dense block
        int minCFind = (int)Math.ceil(density);

        // minimum core number where we should look at
        final int startC = Math.min(minCReorder, minCFind);

        // number of new attributes in the inserted entry
        int newAttNum = numNewAttValues(newEntry);

        //smallest core number higher or equal to startC
        int closest = Integer.MAX_VALUE;
        if(minCReorder < coreNumberToFirstColAndMass.size() && coreNumberToFirstColAndMass.containsKey(minCReorder)) { // small minC
            for(int coreNumber = 0; coreNumber <= minCReorder; coreNumber++) {
                if(coreNumberToFirstColAndMass.containsKey(coreNumber)) {
                    Pair<TableCol, long[]> firstColAndMass = coreNumberToFirstColAndMass.get(coreNumber);
                    long[] massAndRemainedNum = firstColAndMass.getValue();
                    massAndRemainedNum[0] += value;
                    massAndRemainedNum[1] += newAttNum;
                    if (coreNumber >= startC && coreNumber < closest) {
                        closest = coreNumber;
                    }
                }
            }
        }
        else {
            for (int coreNumber : coreNumberToFirstColAndMass.keySet()) {
                if (coreNumber <= minCReorder) {
                    Pair<TableCol, long[]> firstColAndMass = coreNumberToFirstColAndMass.get(coreNumber);
                    long[] massAndRemainedNum = firstColAndMass.getValue();
                    massAndRemainedNum[0] += value;
                    massAndRemainedNum[1] += newAttNum;
                }
                if (coreNumber >= startC && coreNumber < closest) {
                    closest = coreNumber;
                }
            }
        }

        // column of the table where reordered columns should be appended
        TableCol head = null;
        final Pair<TableCol, long[]> firstColAndMassNum = coreNumberToFirstColAndMass.get(closest);

        //set initial value of mass and remained attribute num
        if(firstColAndMassNum != null) {
            head = firstColAndMassNum.getKey().prev;
            long[] massAndRemainedNum = firstColAndMassNum.getValue();
            mass = massAndRemainedNum[0];
            remainedNum = massAndRemainedNum[1];

            if(closest > minCReorder) {
                mass += value;
                remainedNum += newAttNum;
            }
        }

        // first column we should look at
        TableCol col;
        if(head == null) {
            col = table.head;
        }
        else {
            col = head.next;
        }

        // look at attribute values where the dense block may appear
        while(col != null) {

            if(col.coreNumber >= minCReorder) { // inserted attribute value (v_{f}) is found
                break;
            }
            
            if(minCFind <= col.removeMass) {
                currentDensity = ((double)mass) / remainedNum;
                if(currentDensity > density) {
                    density = currentDensity;
                    minCFind = (int)Math.ceil(density);
                    maxMode = col.mode;
                    maxAttVal = col.attVal;
                }
            }
            mass -= col.removeMass;
            remainedNum--;
            col = col.next;
        }

        if(col != null) {
            head = col.prev;
        } else {
            if(table.tail != null) { // reach the last column
                head = table.tail;
            }
        }

        int[] modeToReorderedAttNum = null;
        if(minCReorder < maxCReorder) { 

            final int[] modeToReorderedAttNumWide = new int[order];
            int maxRemovedMass = 0;

            while(true) {
                if(newEntry[col.mode] == col.attVal) { // inserted attribute value (v_{f}) is found
                    head = col.prev;
                    maxRemovedMass = col.removeMass + value;
                    break;
                }
                else {
                    int dim = col.mode;
                    int attVal = col.attVal;
                    modeToAttValsInWideRange[dim][modeToReorderedAttNumWide[dim]++] = attVal;
                    modeToAttValToStatus[dim][attVal] = 1; // before wide range
                }
                
                if(minCFind <= col.removeMass) {
                    currentDensity = ((double) mass) / remainedNum;
                    if (currentDensity > density) {
                        density = currentDensity;
                        minCFind = (int)Math.ceil(density);
                        maxMode = col.mode;
                        maxAttVal = col.attVal;
                    }
                }
                
                mass -= col.removeMass;
                remainedNum--;
                col = col.next;
            }

            TableCol tempCol = col;
            while(tempCol != null) {
                if(tempCol.removeMass >= maxRemovedMass) {
                    break;
                }
                else{
                    int dim = tempCol.mode;
                    int attVal = tempCol.attVal;
                    modeToAttValsInWideRange[dim][modeToReorderedAttNumWide[dim]++] = attVal;
                    modeToAttValToStatus[dim][attVal] = 2; //in wide range
                }
                tempCol= tempCol.next;
            }

            for(int dim = 0; dim < order; dim++) {
                if(table.modeToAttValToCol[dim][newEntry[dim]]==null) {
                    int attVal = newEntry[dim];
                    TableCol newCol = new TableCol(dim, attVal, 0, minCReorder); // core number is set to value;
                    table.addToHead(newCol);
                    modeToAttValsInWideRange[dim][modeToReorderedAttNumWide[dim]++] = attVal;
                    modeToAttValToStatus[dim][attVal] = 2; //in wide range
                }
            }

            final Queue<int[]> seeds = new LinkedList<int[]>();
            seeds.add(new int[]{col.mode, col.attVal});
            newEntry[indexForProcessed] = 1;
            modeToReorderedAttNum = composeSubTensor(seeds, minCReorder, maxCReorder);

            for (int dim = 0; dim < order; dim++) {
                int attNum = modeToReorderedAttNumWide[dim];
                int[] indexToRemovedList = modeToAttValsInWideRange[dim];
                byte[] attValToRemoved = modeToAttValToStatus[dim];
                for(int i=0; i<attNum; i++) {
                    attValToRemoved[indexToRemovedList[i]] = 0; // initialize status;
                }
            }

            if(head != null) {
                col = head.next;
            }
            else {
                col = table.head;
            }
        }
        else { // minCReorder == maxCReorder

            newEntry[indexForProcessed] = 0;

            modeToReorderedAttNum = new int[order];
            final TableCol[][] modeToAttValToCol = table.modeToAttValToCol;
            final boolean[] insertFlag = new boolean[order];
            for(int dim = 0; dim < order; dim++) {
                int attVal = newEntry[dim];
                if(modeToAttValToCol[dim][attVal] == null) {
                    modeToAttValsInNarrowRange[dim][modeToReorderedAttNum[dim]++] = attVal;
                    insertFlag[dim] = true;
                }
            }
            subTensor.insert(newEntry, insertFlag);
        }

        //initialize min heap
        final int[][][][] modeToAttValToEntries = subTensor.modeToAttValToEntries;
        final int[][] modeToAttValToDegree = subTensor.modeToAttValToDegree;
        final int[][] modeToAttValToCardinality = subTensor.modeToAttValToCardinality;

        //initialize hash map
        int minHeapSizeSum = 0;

        for(int dim = 0; dim < order; dim++) {
            HashIndexedMinHeap minHeap = modeToMinHeap[dim];
            int attNum = modeToReorderedAttNum[dim];
            int[] indexToAttVal = modeToAttValsInNarrowRange[dim];
            for(int i=0; i<attNum; i++) {
                int attVal = indexToAttVal[i];
                minHeap.insert(attVal, modeToAttValToDegree[dim][attVal]);
            }
            minHeapSizeSum += attNum;
        }

        // initial core number
        int coreNumber = (head == null) ? -1 : head.coreNumber;

        //while all the entries are reordered
        TableCol newCol = null;
        final TableCol[][] modeToAttValToCol = table.modeToAttValToCol;
        final int[][] modeToAttValToCoreNumber = table.modeToAttValToCoreNumber;
        
        while(minHeapSizeSum > 0) {

            int minHeapMass = Integer.MAX_VALUE;
            int minDim = 0;
            for(int dim = 0; dim < order; dim++) {
                int[] pair = modeToMinHeap[dim].peek();
                if(pair != null && pair[1] < minHeapMass) {
                    minDim = dim;
                    minHeapMass = pair[1];
                }
            }

            while(col!= null && minHeapMass > col.removeMass) {

                int removeMass = col.removeMass;
                if(removeMass > coreNumber) {
                    for(int key = coreNumber+1; key < removeMass; key++) {
                        coreNumberToFirstColAndMass.remove(key);
                    }
                    coreNumberToFirstColAndMass.put(removeMass, new Pair(col, new long[]{mass, remainedNum}));
                    coreNumber = removeMass;
                }
                if(minCFind <= removeMass) {
                    currentDensity = ((double) mass) / remainedNum;
                    if(currentDensity > density) {
                        density = currentDensity;
                        minCFind = (int)Math.ceil(density);
                        maxMode = col.mode;
                        maxAttVal = col.attVal;

                    }
                }
                mass -= removeMass;
                remainedNum--;
                
                if(head != null) {
                    head.next = col;
                } else {
                    table.head = col;
                }

                col.prev = head;
                head = col;
                col = col.next;

            }

            // pop an attribute value from heap
            final int[] pair = modeToMinHeap[minDim].poll();
            final int dim = minDim;
            final int attVal = pair[0];
            final int removeMass = pair[1];
            minHeapSizeSum--;

            newCol = new TableCol(dim, attVal, removeMass, coreNumber); //, mass, density);
            modeToAttValToCoreNumber[dim][attVal] = coreNumber;
            if(removeMass > coreNumber) {
                newCol.coreNumber = removeMass;
                modeToAttValToCoreNumber[dim][attVal] = removeMass;
                for(int key = coreNumber+1; key < removeMass; key++) {
                    coreNumberToFirstColAndMass.remove(key);
                }
                coreNumberToFirstColAndMass.put(removeMass, new Pair(newCol, new long[]{mass, remainedNum}));
                coreNumber = removeMass;
            }

            if(minCFind <= removeMass) {
                currentDensity = ((double) mass) / remainedNum;
                if (currentDensity > density) {
                    density = currentDensity;
                    minCFind = (int)Math.ceil(density);
                    maxMode = dim;
                    maxAttVal = attVal;
                }
            }
            mass -= removeMass;
            remainedNum--;

            // create a new column
            table.modeToAttValToCol[dim][attVal] = newCol;
            if(head != null) {
                head.next = newCol;
            } else {
                table.head = newCol;
            }
            newCol.prev = head;
            head = newCol;

            //update the degree of other attribute values
            int cardinality = modeToAttValToCardinality[dim][attVal];
            int[][] indexToEntry = modeToAttValToEntries[dim][attVal];
            for(int i=0; i<cardinality; i++) {
                int[] entry = indexToEntry[i];
                if(entry[indexForProcessed] == 0) { // if this entry is not removed yet
                    for (int _dim = 0; _dim < order; _dim++) {
                        if (_dim != dim) {
                            int key = entry[_dim];
                            HashIndexedMinHeap minHeap = modeToMinHeap[_dim];
                            if (minHeap.containsKey(key)) {
                                minHeap.refreshPriority(key, minHeap.getPriority(key) - entry[order]);
                            }
                        }
                    }
                    entry[indexForProcessed] =1; // this entry is removed
                }
            }

            subTensor.deleteAttVal(dim, attVal);
        }

        if(col != null) {
            head.next = col;
            col.prev = head;
        } else { 
            table.tail = newCol;
        }

        // update the maintained block if we found a denser one
        if(density > prevMaxDensity) {

            isBlockChanged = true;

            if(maintainedBlock !=null) {
                for (int dim = 0; dim < order; dim++) {
                    for (int attVal : maintainedBlock[dim]) {
                        modeToAttValToInMaintained[dim][attVal] = false;
                    }
                }
            }

            if(maxMode != -1) {

                maintainedBlock = new List[order];
                maintainedAttNum = 0;
                for (int dim = 0; dim < order; dim++) {
                    maintainedBlock[dim] = new LinkedList();
                }
                col = modeToAttValToCol[maxMode][maxAttVal];
                while (col != null) {
                    maintainedBlock[col.mode].add(col.attVal);
                    modeToAttValToInMaintained[col.mode][col.attVal] = true;
                    col = col.next;
                    maintainedAttNum++;
                }
            }
            else {
                maintainedBlock = null;
                maintainedAttNum = -1;
            }
        }
    }

    /**
     * delete an entry (or decrement the value if exists) and updated the densest block
     * @param deletedEntry
     * @return modeToRemoved mode to whether attribute value is removed
     */
    boolean[] delete(int[] deletedEntry) {

        deletedEntry = deletedEntry.clone();

        if (deletedEntry[order] == 0 || oriTensor == null) {
            return null;
        }

        if(oriTensor.cardinality == 0) {
            System.out.println("Deletion failed: the input tensor is empty");
            density = 0;
            maintainedAttNum = -1;
            return null;
        }

        // delete entry
        final int[] modeToNewLength = oriTensor.delete(deletedEntry);

        if(modeToNewLength==null) { //ignore
            System.out.println("Deletion failed: an unknown entry");
            return null;
        }

        // number of attribute values deleted in the deleted entry
        int deletedAttNum = 0;

        // number of attribute values (1) in the deleted entry and (2) belonging to the maintained block
        int maintainedNum = 0;

        boolean[] modeToRemoved = new boolean[order];
        for(int dim = 0; dim < order; dim++) {
            if(modeToNewLength[dim] > 0) {
                subTensor.resize(dim, deletedEntry[dim], modeToNewLength[dim]);
            }
            else if(modeToNewLength[dim] < 0) { //remove
                modeToRemoved[dim] = true;
                deletedAttNum++;
            }
            if(modeToAttValToInMaintained[dim][deletedEntry[dim]])
                maintainedNum++;
        }

        final int value = deletedEntry[order];

        final int[] minMaxC = findDeleteMinMaxCoreNum(deletedEntry, modeToRemoved);

        // minimum core number we should look at for reordering
        final int minCReorder = minMaxC[0];

        // maximum core number we should look at for reordering
        final int maxCReorder = minMaxC[1];

        // current mass
        long mass = oriTensor.mass;

        // number of remained attribute values
        long remainedNum = oriTensor.cardinality;

        boolean isDensestAttUpdated = false;
        
        // whether maintained block needs to be updated
        final boolean isMaintainedUpdated = maintainedNum == order && maintainedAttNum != -1;
        if(isMaintainedUpdated && maintainedAttNum > deletedAttNum) {
            density = (density * maintainedAttNum - value) / (maintainedAttNum - deletedAttNum);
        }
        else if (isMaintainedUpdated) {
            density = oriTensor.cardinality == 0 ? 0 : oriTensor.mass / oriTensor.cardinality;
            maxMode = -1;
        }

        //minimum core number we should look at for finding a dense block
        int minCFind = (int) Math.ceil(density);
        int startC = Math.min(minCFind, minCReorder);
        if(!isMaintainedUpdated) {
            startC = minCReorder;
        }

        if(minCReorder == maxCReorder) { // simply remove
            for(int dim = 0; dim < order; dim++) {
                if(modeToRemoved[dim]) { // removed
                    TableCol colToDelete = table.modeToAttValToCol[dim][deletedEntry[dim]];
                    Pair<TableCol, long[]> colAndMasses = coreNumberToFirstColAndMass.get(colToDelete.coreNumber);
                    if(colToDelete == colAndMasses.getKey()) {
                        coreNumberToFirstColAndMass.remove(colToDelete.coreNumber);
                        if(colToDelete.next != null && colToDelete.next.coreNumber == colToDelete.coreNumber) {
                            coreNumberToFirstColAndMass.put(colToDelete.next.coreNumber, new Pair<TableCol, long[]>(colToDelete.next, colAndMasses.getValue()));
                        }
                    }
                    table.delete(colToDelete);
                }
            }
        }

        //smallest core number higher or equal to startC
        int closest = Integer.MAX_VALUE;
        if(minCReorder < coreNumberToFirstColAndMass.size() && coreNumberToFirstColAndMass.containsKey(minCReorder)) { // small minC
            for(int coreNumber = 0; coreNumber < minCReorder; coreNumber++) {
                if(coreNumberToFirstColAndMass.containsKey(coreNumber)) {
                    Pair<TableCol, long[]> firstColAndMass = coreNumberToFirstColAndMass.get(coreNumber);
                    long[] massAndRemainedNum = firstColAndMass.getValue();
                    massAndRemainedNum[0] -= value;
                    massAndRemainedNum[1] -= deletedAttNum;
                    if (coreNumber >= startC && coreNumber < closest) {
                        closest = coreNumber;
                    }
                }
            }
            if(closest > minCReorder) {
                closest = minCReorder;
            }
        }
        else {
            for (int coreNumber : coreNumberToFirstColAndMass.keySet()) {
                if (coreNumber < minCReorder) {
                    Pair<TableCol, long[]> firstColAndMass = coreNumberToFirstColAndMass.get(coreNumber);
                    long[] massAndRemainedNum = firstColAndMass.getValue();
                    massAndRemainedNum[0] -= value;
                    massAndRemainedNum[1] -= deletedAttNum;
                }
                if (coreNumber >= startC && coreNumber < closest) {
                    closest = coreNumber;
                }
            }
        }


        TableCol head = null;
        final Pair<TableCol, long[]> firstColAndMassNum = coreNumberToFirstColAndMass.get(closest);

        // first column we should look at
        TableCol col = null;

        //set initial value of mass and remainedNum
        if(firstColAndMassNum != null) {
            col = firstColAndMassNum.getKey();
            head = firstColAndMassNum.getKey().prev;
            long[] massAndRemainedNum = firstColAndMassNum.getValue();
            mass = massAndRemainedNum[0];
            remainedNum = massAndRemainedNum[1];
            if (closest >= minCReorder) {
                mass -= value;
                remainedNum -= deletedAttNum;
            }
        }
        else {
            head = table.tail;
        }

        double currentDensity;

        // look at attribute values where the densest block may appear
        while(col != null) {

            if(col.coreNumber >= minCReorder) { // inserted attribute value is found
                break;
            }

            // remove attribute value corresponding to the current column
            if(minCFind <= col.removeMass) {
                currentDensity = ((double)mass) / remainedNum;
                if(currentDensity > density) {
                    density = currentDensity;
                    minCFind = (int)Math.ceil(density);
                    maxMode = col.mode;
                    maxAttVal = col.attVal;
                    isDensestAttUpdated = true;
//                    System.out.println("maxAverageMass0: " + averageMass + "," + mass + "," + remainedNum + "," + col.mode + "," + col.attVal); // debug
                }
            }
            mass -= col.removeMass;
            remainedNum--;
            col = col.next;
        }

        if(col != null) {
            head = col.prev;
        } else {
            if(table.tail != null) { //reach the last column
                head = table.tail;
            }
        }

        if(minCReorder != maxCReorder) {

            final int[] modeToReorderedAttNumWide = new int[order];
            TableCol tempCol = col;
            boolean isDeletedEntryFound = false;
            int minRemoveMass = 0;

            while(tempCol != null) {

                if(!isDeletedEntryFound && deletedEntry[tempCol.mode] == tempCol.attVal ) { // inserted attribute value is found
                    isDeletedEntryFound = true;
                    minRemoveMass = tempCol.coreNumber;
                }
                else if (isDeletedEntryFound && tempCol.removeMass >= minRemoveMass) {
                    break;
                }

                int dim = tempCol.mode;
                int attVal = tempCol.attVal;
                modeToAttValsInWideRange[dim][modeToReorderedAttNumWide[dim]++] = attVal;
                modeToAttValToStatus[dim][attVal] = 2; //in wide range
                tempCol= tempCol.next;
            }

            //remove entries
            for (int dim = 0; dim < order; dim++) {
                if (modeToRemoved[dim]) { // removed
                    table.delete(table.modeToAttValToCol[dim][deletedEntry[dim]]);
                }
            }

            final Queue<int[]> seeds = new LinkedList();
            for(int dim = 0; dim < order; dim++) {
                if(!modeToRemoved[dim]) {
                    if(modeToAttValToStatus[dim][deletedEntry[dim]] == 2) { // in range
                        seeds.add(new int[]{dim, deletedEntry[dim]});
                    }
                }
            }

            final int[] modeToReorderedAttNum = composeSubTensor(seeds, minCReorder, maxCReorder);

            for (int dim = 0; dim < order; dim++) {
                int attNum = modeToReorderedAttNumWide[dim];
                int[] indexToRemovedList = modeToAttValsInWideRange[dim];
                byte[] attValToRemoved = modeToAttValToStatus[dim];
                for(int i=0; i<attNum; i++) {
                    attValToRemoved[indexToRemovedList[i]] = 0; // undo;
                }
            }

            //initialize min heap
            final int[][][][] modeToAttValToEntries = subTensor.modeToAttValToEntries;
            final int[][] modeToAttValToDegree = subTensor.modeToAttValToDegree;
            final int[][] modeToAttValToCardinality = subTensor.modeToAttValToCardinality;

            //initialize hash map
            int minHeapSizeSum = 0;

            for (int dim = 0; dim < order; dim++) {
                HashIndexedMinHeap minHeap = modeToMinHeap[dim];
                int attNum = modeToReorderedAttNum[dim];
                int[] indexToAttVal = modeToAttValsInNarrowRange[dim];
                for (int i = 0; i < attNum; i++) {
                    int attVal = indexToAttVal[i];
                    minHeap.insert(attVal, modeToAttValToDegree[dim][attVal]);
                }
                minHeapSizeSum += attNum;
            }

            if(head != null) {
                col = head.next;
            }
            else {
                col = table.head;
            }

            // initial core number
            int coreNumber = (head == null) ? -1 : head.coreNumber;

            //while all the entries are reordered
            TableCol newCol = null;
            final TableCol[][] modeToAttValToCol = table.modeToAttValToCol;
            final int[][] modeToAttValToCoreNumber = table.modeToAttValToCoreNumber;

            while (minHeapSizeSum > 0) {

                int minHeapMass = Integer.MAX_VALUE;
                int minDim = 0;
                for (int dim = 0; dim < order; dim++) {
                    int[] pair = modeToMinHeap[dim].peek();
                    if (pair != null && pair[1] < minHeapMass) {
                        minDim = dim;
                        minHeapMass = pair[1];
                    }
                }

                while (col != null && minHeapMass > col.removeMass) {

                    // remove attribute value corresponding to this entry
                    int removeMass = col.removeMass;
                    if (removeMass > coreNumber) {
                        for (int key = coreNumber + 1; key < removeMass; key++) {
                            coreNumberToFirstColAndMass.remove(key);
                        }
                        coreNumberToFirstColAndMass.put(removeMass, new Pair(col, new long[]{mass, remainedNum}));
                        coreNumber = removeMass;
                    }
                    if(isMaintainedUpdated && minCFind <= removeMass) {
                        currentDensity = ((double) mass) / remainedNum;
                        if(currentDensity >= density) {
                            density = currentDensity;
                            minCFind = (int)Math.ceil(density);
                            maxMode = col.mode;
                            maxAttVal = col.attVal;
                            isDensestAttUpdated = true;
                      }
                    }
                    mass -= removeMass;
                    remainedNum--;

                    // store current entry
                    if (head != null) {
                        head.next = col;
                    } else {
                        table.head = col;
                    }

                    col.prev = head;
                    head = col;
                    col = col.next;

                }

                // pop an attribute-value from heap
                final int[] pair = modeToMinHeap[minDim].poll();
                final int dim = minDim;
                final int attVal = pair[0];
                final int removeMass = pair[1];
                minHeapSizeSum--;

                newCol = new TableCol(dim, attVal, removeMass, coreNumber);
                modeToAttValToCoreNumber[dim][attVal] = coreNumber;
                if (removeMass > coreNumber) {
                    newCol.coreNumber = removeMass;
                    modeToAttValToCoreNumber[dim][attVal] = removeMass;
                    for (int key = coreNumber + 1; key < removeMass; key++) {
                        coreNumberToFirstColAndMass.remove(key);
                    }
                    coreNumberToFirstColAndMass.put(removeMass, new Pair(newCol, new long[]{mass, remainedNum}));
                    coreNumber = removeMass;
                }
                if(isMaintainedUpdated && minCFind <= removeMass) {
                    currentDensity = ((double) mass) / remainedNum;
                    if (currentDensity >= density) {
                        density = currentDensity;
                        minCFind = (int)Math.ceil(density);
                        maxMode = dim;
                        maxAttVal = attVal;
                        isDensestAttUpdated = true;
                    }
                }
                mass -= removeMass;
                remainedNum--;

                // create a new column
                modeToAttValToCol[dim][attVal] = newCol;
                if (head != null) {
                    head.next = newCol;
                } else {
                    table.head = newCol;
                }
                newCol.prev = head;
                head = newCol;

                //update the degree of other attribute values
                int cardinality = modeToAttValToCardinality[dim][attVal];
                int[][] indexToEntry = modeToAttValToEntries[dim][attVal];
                for (int i = 0; i < cardinality; i++) {
                    int[] entry = indexToEntry[i];
                    if (entry[indexForProcessed] == 0) {
                        for (int _dim = 0; _dim < order; _dim++) {
                            if (_dim != dim) {
                                int key = entry[_dim];
                                HashIndexedMinHeap minHeap = modeToMinHeap[_dim];
                                if (minHeap.containsKey(key)) {
                                    minHeap.refreshPriority(key, minHeap.getPriority(key) - entry[order]);
                                }
                            }
                        }
                        entry[indexForProcessed] = 1; // this entry is removed
                    }
                }
                subTensor.deleteAttVal(dim, attVal);
            }

            if(head != null) {
                if (col != null) {
                    head.next = col;
                    col.prev = head;
                } else { // last column
                    table.tail = head;
                }
            }
            else {
                table.head = col;
                col.prev = head;
            }

            if(coreNumber < maxCReorder - 1) {
                while (col != null) {
                    // remove attribute value corresponding to the current column
                    int removeMass = col.removeMass;
                    if (removeMass > coreNumber) {
                        for (int key = coreNumber + 1; key < removeMass; key++) {
                            coreNumberToFirstColAndMass.remove(key);
                        }
                        coreNumberToFirstColAndMass.put(removeMass, new Pair(col, new long[]{mass, remainedNum}));
                        coreNumber = removeMass;
                        if(removeMass >= maxCReorder -1) {
                            break;
                        }
                    }
                    if (isMaintainedUpdated && minCFind <= col.removeMass) {
                        currentDensity = ((double) mass) / remainedNum;
                        if (currentDensity >= density) {
                            density = currentDensity;
                            minCFind = (int) Math.ceil(density);
                            maxMode = col.mode;
                            maxAttVal = col.attVal;
                            isDensestAttUpdated = true;
                        }
                    }
                    mass -= col.removeMass;
                    remainedNum--;
                    col = col.next;
                }
            }

            if(col==null) { //last column
                List<Integer> keyToRemove = new LinkedList<Integer>();
                for (int key : coreNumberToFirstColAndMass.keySet()) {
                    if (coreNumber < key) {
                        keyToRemove.add(key);
                    }
                }
                for(int key : keyToRemove) {
                    coreNumberToFirstColAndMass.remove(key);
                }
            }
        }

        if(isMaintainedUpdated) {
            while (col != null) {
                // remove attribute value corresponding to the current column
                if (minCFind <= col.removeMass) {
                    currentDensity = ((double) mass) / remainedNum;
                    if (currentDensity >= density) {
                        density = currentDensity;
                        minCFind = (int) Math.ceil(density);
                        maxMode = col.mode;
                        maxAttVal = col.attVal;
                        isDensestAttUpdated = true;
                    }
                }
                mass -= col.removeMass;
                remainedNum--;
                col = col.next;
            }
        }

        if(maxMode == -1) { // entire block is the maintained block
            density =  oriTensor.cardinality == 0 ? 0 : ((double)oriTensor.mass) / oriTensor.cardinality;
            maintainedAttNum = -1;
            if(deletedAttNum > 0) {
                isBlockChanged = true;
            }
            return modeToRemoved;
        }
        else if(isMaintainedUpdated) { // entry is included in the maintained block

            isBlockChanged = true;

            if(!isDensestAttUpdated) { // maintained block is composed by the same attributes
                if(deletedAttNum > 0) {
                    for (int dim = 0; dim < order; dim++) {
                        if (modeToRemoved[dim]) {
                            modeToAttValToInMaintained[dim][deletedEntry[dim]] = false;
                            maintainedBlock[dim].remove((Object) deletedEntry[dim]);
                            maintainedAttNum -= 1;
                        }
                    }
                }
            }
            else {
                if (maintainedBlock != null) {
                    for (int dim = 0; dim < order; dim++) {
                        for (int attVal : maintainedBlock[dim]) {
                            modeToAttValToInMaintained[dim][attVal] = false;
                        }
                    }
                }

                if (maxMode != -1) {
                    final TableCol[][] modeToAttValToCol = table.modeToAttValToCol;
                    maintainedBlock = new List[order];
                    maintainedAttNum = 0;
                    for (int dim = 0; dim < order; dim++) {
                        maintainedBlock[dim] = new LinkedList();
                    }
                    col = modeToAttValToCol[maxMode][maxAttVal];
                    while (col != null) {
                        maintainedBlock[col.mode].add(col.attVal);
                        modeToAttValToInMaintained[col.mode][col.attVal] = true;
                        col = col.next;
                        maintainedAttNum++;
                    }
                } else {
                    maintainedBlock = null;
                    maintainedAttNum = -1;
                }

            }

        }

        return modeToRemoved;
    }

    /**
     * return a subtensor consisting of attribute values whose location can be reordered
     * @param seeds
     * @param minCoreNum
     * @param maxCoreNum
     * @return
     */
    private int[] composeSubTensor(Queue<int[]> seeds, int minCoreNum, int maxCoreNum) {

        final int[][][][] modeToAttValToEntries = oriTensor.modeToAttValToEntries; //original oriTensor entries
        final int[][] modeToAttValToCardinality = oriTensor.modeToAttValToCardinality;
        final int[] modeToReorderNum = new int[order];

        final TableCol[][] modeToAttValToCol = table.modeToAttValToCol;
        final int[][] modeToAttValToCoreNumber = table.modeToAttValToCoreNumber;

        final Queue<int[]> queue = seeds; // list of seeds
        for(int[] seed : seeds) {
            modeToAttValToStatus[seed[0]][seed[1]] = 3; // in the narrow range
        }

        //until queue is empty
        final int[] modeToCoreNumber = new int[order];
        final boolean[] insertFlag = new boolean[order];
        while(!queue.isEmpty()) {
            final int[] pair = queue.poll();
            final int seedDim = pair[0];
            final int seedVal = pair[1];
            table.delete(modeToAttValToCol[seedDim][seedVal]);
            modeToAttValsInNarrowRange[seedDim][modeToReorderNum[seedDim]++] = seedVal;

            final int cardinality = modeToAttValToCardinality[seedDim][seedVal];
            final int[][] indexToEntry = modeToAttValToEntries[seedDim][seedVal];
            out:for(int i=0; i<cardinality; i++) {
                int[] entry = indexToEntry[i];
                if(entry[indexForProcessed] == 0) { //already added to the subtensor
                    continue;
                }


                for (int entryDim = 0; entryDim < order; entryDim++) {
                    if (seedDim != entryDim) {
                        modeToCoreNumber[entryDim] = modeToAttValToCoreNumber[entryDim][entry[entryDim]];
                        if(modeToCoreNumber[entryDim] < minCoreNum ||
                                (modeToCoreNumber[entryDim] == minCoreNum && modeToAttValToStatus[entryDim][entry[entryDim]] == 1)) { //removed
                            continue out;
                        }
                    }
                }

                boolean isInsertOne = true;
                for(int entryDim = 0; entryDim < order; entryDim++) {
                    if(seedDim != entryDim) {
                        if(modeToCoreNumber[entryDim] >= minCoreNum
                                && modeToCoreNumber[entryDim] < maxCoreNum && modeToAttValToStatus[entryDim][entry[entryDim]] >= 2){ // appropriate core number
                            isInsertOne = false;
                            insertFlag[entryDim] = true;
                            if(modeToAttValToStatus[entryDim][entry[entryDim]] != 3) { // not added to the queue yet
                                queue.add(new int[]{entryDim, entry[entryDim]});
                                modeToAttValToStatus[entryDim][entry[entryDim]] = 3;
                            }
                        }
                        else {
                            insertFlag[entryDim] = false;
                        }
                    }
                }

                if(isInsertOne) {
                    entry[indexForProcessed] = 1;
                    subTensor.addDegree(entry, seedDim);
                }
                else {
                    entry[indexForProcessed] = 0;
                    insertFlag[seedDim] = true;
                    subTensor.insert(entry, insertFlag);
                }

            }

        }

        return modeToReorderNum;
    }

    /**
     * return minimum and maximum core number we should look at for reordering for insertion
     * @param entry
     * @return
     */
    private int[] findMinMaxCoreNumberForInsertion(int[] entry) {
        int minCoreNum = Integer.MAX_VALUE;
        int maxCoreNum = Integer.MAX_VALUE;
        for(int dim = 0; dim < order; dim++) {
            if(table.modeToAttValToCol[dim][entry[dim]]!=null) {
                TableCol col = table.modeToAttValToCol[dim][entry[dim]];
                if(minCoreNum > col.coreNumber) {
                    minCoreNum = col.coreNumber;
                }
                int attMaxCoreNum = col.coreNumber+entry[order];
                if(maxCoreNum > attMaxCoreNum) {
                    maxCoreNum = attMaxCoreNum;
                }
            } else {
                int value = entry[order];
                if (minCoreNum > value) {
                    minCoreNum = value;
                }
                if (maxCoreNum > value) {
                    maxCoreNum = value;
                }
            }
        }
        return new int[]{minCoreNum, maxCoreNum};
    }

    /**
     * return minimum and maximum core number we should look at for reordering for deletion
     * @param removedEntry
     * @return
     */
    private int[] findDeleteMinMaxCoreNum(final int[] removedEntry, final boolean[] modeToRemoved) {

        final int[][] modeToAttValToCoreNumber = table.modeToAttValToCoreNumber;
        int curMinCoreNum = Integer.MAX_VALUE; // current minimum core number
        for(int dim = 0; dim < order; dim++) {
            int coreNum = modeToAttValToCoreNumber[dim][removedEntry[dim]];
            if(curMinCoreNum > coreNum) {
                curMinCoreNum = coreNum;
            }
        }

        int minCoreNum = curMinCoreNum + 1;
        int value = removedEntry[order];

        for(int dim =0; dim < order; dim++) {
            int currentCoreNum = table.modeToAttValToCoreNumber[dim][removedEntry[dim]];
            if(currentCoreNum != curMinCoreNum || modeToRemoved[dim]) { //skip if it is removed
                continue;
            }
            int newCoreNum = curMinCoreNum + 1 - value;
            if(minCoreNum > newCoreNum) {
                minCoreNum = newCoreNum;
            }
        }
        return new int[]{minCoreNum, curMinCoreNum + 1};
    }

    /**
     * number of new attribute values in the entry
     * @param entry
     * @return
     */
    private int numNewAttValues(int[] entry) {
        int numOfNewCols = 0;
        for(int dim = 0; dim < order; dim++) {
            if(table.modeToAttValToCol[dim][entry[dim]]==null) {
                numOfNewCols++;
            }
        }
        return numOfNewCols;
    }

    /**
     * create a tensor (minimal feature) from the given tensor (full feature)
     * @param tensor
     * @param modeToIndicesNum
     * @return
     */
    private TensorMinimal createTensorWithSameSize(TensorFull tensor, int[] modeToIndicesNum) {
        final TensorMinimal newTensor = new TensorMinimal(tensor.order, modeToIndicesNum);
        for(int dim = 0; dim < tensor.order; dim++) {
            int[][][] attValToEntryOld = tensor.modeToAttValToEntries[dim];
            int[][][] attValToEntryNew = newTensor.modeToAttValToEntries[dim];
            int attValNum = attValToEntryOld.length;
            for(int attVal = 0; attVal < attValNum; attVal++) {
                if(attValToEntryOld[attVal]!=null) {
                    attValToEntryNew[attVal] = new int[attValToEntryOld[attVal].length][];
                }
            }
        }
        return newTensor;
    }

}
