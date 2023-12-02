import java.io.*;

public class ModelManager {

    public static void saveModel(NeuralNetwork network, String filename) throws IOException {

        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename))) {

            out.writeObject(network);
        }
    }

    public static NeuralNetwork loadModel(String filename) throws IOException, ClassNotFoundException {

        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename))) {

            return (NeuralNetwork) in.readObject();
        }
    }
}
