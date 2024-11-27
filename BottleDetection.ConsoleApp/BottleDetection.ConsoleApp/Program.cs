using System.Text.RegularExpressions;
using Compunet.YoloSharp;
using Compunet.YoloSharp.Plotting;
using SixLabors.ImageSharp;

namespace BottleDetection.ConsoleApp
{
    public class Program
    {

        public static string outputDir = "C:/Users/8376/Pictures/Output";
        public static string inputDir = "C:/Users/8376/OneDrive - OurResource/Pictures/Camera Roll";

        public static async Task Main(string[] args) { 

            // Load the YOLO predictor
            using var predictor = new YoloPredictor("weights/best.onnx");

            var images = GetImagesFromDir();

            await ProcessImages(predictor, images);
        }

        private static async Task ProcessImages(YoloPredictor predictor, List<string> images)
        {
            foreach (var image in images)
            {
                await ProcessImage(predictor, image);
            }
        }

        private static List<string> GetImagesFromDir()
        {
            var files = Directory.GetFiles(inputDir, "*.*", SearchOption.AllDirectories);

            List<string> imageFiles = new List<string>();
            foreach (string filename in files)
            {
                if (Regex.IsMatch(filename, @"\.jpg$|\.png$|\.gif$"))
                    imageFiles.Add(filename);
            }

            return imageFiles;
        }

        private static async Task ProcessImage(YoloPredictor predictor, string imagePath)
        {
            // Load the target image
            using var image = Image.Load(imagePath);

            // Run model
            var result = await predictor.PoseAsync(image);

            // Create plotted image from model results
            using var plotted = await result.PlotImageAsync(image);

            // Write the plotted image to file
            plotted.Save(String.Format("{0}/{1}", outputDir, Path.GetFileName(imagePath)));
        }

    }

}