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

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * Example Using DenseAlert
 *
 * @author kijungs
 */
public class DenseAlertExample {

    /**
     * Example Code for Dense Stream
     * @throws IOException
     */
    public static void main(String[] ar) throws IOException {

        /**
         * Initialize DenseStream Object
         */
        final String dataPath = "example_data.txt";
        final int order = 3;
        final int window = 60; // 60 seconds
        final String delim = ",";
        final DenseAlert denseAlert = new DenseAlert(order, window);

        BufferedReader br = new BufferedReader(new FileReader(dataPath));

        int count = 0;
        while(true) {

            final String line = br.readLine();
            if(line == null) {
                break;
            }

            denseAlert.insert(parseEntry(line, delim, order), parseTime(line, delim, order));

            if((++count) % 100000 == 0) {
                System.out.println("Current Unix Time: " + parseTime(line, delim, order) + ", Current Density: " + denseAlert.getDensity());
            }
        }

        br.close();

        return;
    }

    private static int[] parseEntry(String line, String delim, int order) {

        String[] tokens = line.split(delim);

        int[] entry = new int[order+1];

        for(int mode = 0; mode < order; mode++) {
            entry[mode] = Integer.valueOf(tokens[mode]);
        }
        entry[order] = Integer.valueOf(tokens[order]);

        return entry;
    }

    private static long parseTime(String line, String delim, int order) {

        String[] tokens = line.split(delim);
        return Long.valueOf(tokens[order+1]);
    }
}
