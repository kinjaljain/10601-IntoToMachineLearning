import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;
import java.io.File;
import java.util.List;
import java.util.Collections;

public class reverse{
	public static void main(String[] args){
		try{
			System.out.println(args.length);
			if(args.length < 2){
				System.out.println("Please give both input and output files to the program");
			}
			String input_filename = args[0];
			String output_filename = args[1];
			System.out.println(input_filename);
			System.out.println(output_filename);
			List<String> lines = Files.readAllLines(Paths.get(input_filename));
			Collections.reverse(lines);
            		Files.write(Paths.get(output_filename), lines);
        	}
		catch(IOException e){
            		e.printStackTrace();
        	} 
	}
}
