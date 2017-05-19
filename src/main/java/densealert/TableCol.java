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
 * A column of a table (mode, attribute value, d_{pi}, c_{pi})
 * @author kijungs
 */
class TableCol {

    public int mode;
    public int attVal;
    public int removeMass; // d_{pi}
    public int coreNumber; // c_{pi}

    public TableCol prev; // previous attribute value in \pi
    public TableCol next; // next attribute value in \pi

    public TableCol(int mode, int attVal, int removeMass, int coreNumber){  //, long remainedMass, double density) {
        this.mode = mode;
        this.attVal = attVal;
        this.removeMass = removeMass;
        this.coreNumber = coreNumber;
    }
}
