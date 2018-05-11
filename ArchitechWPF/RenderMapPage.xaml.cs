using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using YoloWrapper;

namespace ArchitechWPF
{
    /// <summary>
    /// Interaction logic for RenderMapPage.xaml
    /// </summary>
    public partial class RenderMapPage : Page, INotifyPropertyChanged
    {
        public VideoCapture cam { get; set; }
        private BitmapSource img;
        public event PropertyChangedEventHandler PropertyChanged;
        private readonly BackgroundWorker bWorker;
        private string CamIP;
        private string YoloCfgFile;
        private string YoloWeightsFile;
        private string YoloNamesFile;
        private static readonly Scalar[] Colors = Enumerable.Repeat(false, 20).Select(x => Scalar.RandomColor()).ToArray();

        #region opencvmap
        private readonly BackgroundWorker OpenCvWorker;
        private Mat background;
        private Mat binary;
        private int contours;
        #endregion

        public BitmapSource Img
        {
            get
            {
                return img;
            }
            set
            {
                img = value;
                OnPropertyChanged();
            }
        }
        public RenderMapPage(string camIp, string cfg, string weights, string name)
        {
            InitializeComponent();
            this.DataContext = this;

            this.CamIP = camIp;
            this.YoloCfgFile = cfg;
            this.YoloWeightsFile = weights;
            this.YoloNamesFile = name;

            bWorker = new BackgroundWorker();
            bWorker.DoWork += bWorker_DoWork;
            bWorker.RunWorkerCompleted += bWorker_WorkerCompleted;
            bWorker.WorkerReportsProgress = true;
            bWorker.ProgressChanged += bWorker_ProgressChanged;
            #region opencvmap
            OpenCvWorker = new BackgroundWorker();
            OpenCvWorker.DoWork += OpenCvWorker_DoWork;
            OpenCvWorker.RunWorkerCompleted += OpenCvWorker_WorkerCompleted;
            OpenCvWorker.WorkerReportsProgress = true;
            OpenCvWorker.ProgressChanged += OpenCvWorker_ProgressChanged;
            Img = null;
            #endregion
        }

        #region Camera Management
        public void InitCamera()
        {
            cam = new VideoCapture(CamIP);
        }

        public void StopCamera()
        {
            //VideoSource.Stop();
        }

        public void PauseCamera()
        {
            //VideoSource.Pause();
        }
        #endregion

        private void OpenCvWorker_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            FrameControl.Source = OpenCvSharp.Extensions.BitmapSourceConverter.ToBitmapSource(((Mat)e.UserState));
        }

        private void OpenCvWorker_DoWork(object sender, DoWorkEventArgs e)
        {
            InitCamera();
            //Load image
            Mat camImage = cam.RetrieveMat();

            //Convert to gray and apply bilateral filter to smooth the image
            Mat gray = new Mat();
            //Mat camImage = new Mat();
            //Cv2.Resize(camImage, resizedCamImage, new OpenCvSharp.Size(0, 0), 0.3, 0.3);
            Cv2.ImWrite("tmp.jpg", camImage);
            Cv2.CvtColor(camImage, gray, ColorConversionCodes.BGR2GRAY);
            Mat bilateral = new Mat();
            Cv2.BilateralFilter(gray, bilateral, 25, 10, 80);

            //Binarize the image and find the countours of the shape
            Mat useless = new Mat();
            Mat binary = new Mat();
            //Cv2.Threshold(bilateral, binary, 0, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);
            Cv2.Threshold(bilateral, binary, 100, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);
            //Cv2.AdaptiveThreshold(bilateral, binary, 255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.BinaryInv, 11, 2);
            Cv2.FindContours(binary.Clone(), out var contours, useless, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);

            //Create a blank image and draw countours find inside
            Mat background = new Mat(camImage.Rows, camImage.Cols, MatType.CV_8U);

            Mat[] new_contours = new Mat[contours.Length];
            for (int i = 0; i < contours.Length; i++)
            {
               new_contours[i] = DefinePointless(contours[i]);
            }
            Cv2.DrawContours(background, contours, -1, Scalar.White);

            //Invert the image and convert it to RGB
            Mat invertedBackground = new Mat();
            Cv2.BitwiseNot(background, invertedBackground);

            Mat coloredBackground = new Mat();
            Cv2.CvtColor(invertedBackground, coloredBackground, ColorConversionCodes.GRAY2BGR);

            this.background = coloredBackground;
            this.binary = binary;
            this.contours = contours.Length;

            //Report progress
            OpenCvWorker.ReportProgress(100, coloredBackground);
        }

        private Mat DefinePointless(Mat contour)
        {
            List<Vec2i> cont_tmp = new List<Vec2i>();
            for(int i=1; i < contour.Rows; )
            {
                if(Math.Abs(contour.Get<Vec2i>(i).Item0 - contour.Get<Vec2i>(i-1).Item0) == 1 &&
                    Math.Abs(contour.Get<Vec2i>(i).Item1 - contour.Get<Vec2i>(i-1).Item1) == 1)
                {
                    Vec2i vec = new Vec2i();
                    if(contour.Get<Vec2i>(i-1).Item0 < contour.Get<Vec2i>(i).Item0)
                    {
                        if(contour.Get<Vec2i>(i-1).Item1 > contour.Get<Vec2i>(i).Item1)
                        {
                            vec.Item0 = contour.Get<Vec2i>(i - 1).Item0;
                            vec.Item1 = contour.Get<Vec2i>(i).Item1;
                        }
                        else
                        {
                            vec.Item0 = contour.Get<Vec2i>(i).Item0;
                            vec.Item1 = contour.Get<Vec2i>(i - 1).Item1;
                        }
                    }
                    else
                    {
                        if (contour.Get<Vec2i>(i - 1).Item1 > contour.Get<Vec2i>(i).Item1)
                        {
                            vec.Item0 = contour.Get<Vec2i>(i).Item0;
                            vec.Item1 = contour.Get<Vec2i>(i - 1).Item1;
                        }
                        else
                        {
                            vec.Item0 = contour.Get<Vec2i>(i - 1).Item0;
                            vec.Item1 = contour.Get<Vec2i>(i).Item1;
                        }
                    }
                    i += 2;
                    cont_tmp.Add(vec);
                }
                else
                {
                    cont_tmp.Add(contour.Get<Vec2i>(i - 1));
                    if(i+1 == contour.Rows)
                    {
                        cont_tmp.Add(contour.Get<Vec2i>(i));
                    }
                    i++;
                }
            }

            return new Mat(cont_tmp.Count, 1, MatType.CV_32SC2, cont_tmp.ToArray());
        }
        private bool CheckContoursAmount(int actualContoursAmount, Mat binary, OpenCvSharp.Point p1, OpenCvSharp.Point p2)
        {
            Mat tmp_binary = new Mat();
            tmp_binary = binary.Clone();
            Cv2.Rectangle(tmp_binary, p1, p2, Scalar.White, -1);
            Mat useless = new Mat();
            Cv2.FindContours(tmp_binary, out var contours, useless, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);
            return contours.Length > actualContoursAmount;
        }

        private void OpenCvWorker_WorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            bWorker.RunWorkerAsync();
        }

        private void bWorker_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            FrameControl.Source = OpenCvSharp.Extensions.BitmapSourceConverter.ToBitmapSource(((Mat)e.UserState));
        }

        private void bWorker_WorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            //throw new NotImplementedException();
        }

        private void bWorker_DoWork(object sender, DoWorkEventArgs e)
        {

            //Loading YOLO config
            Darknet darknet = new Darknet(YoloCfgFile, YoloWeightsFile);
            var classNames = File.ReadAllLines(YoloNamesFile);
            //Capturing frames
            for (; ; )
            {
                Mat originalBackground = this.background.Clone();
                Mat frame;
                frame = cam.RetrieveMat();

                if (frame.Empty())
                {
                    Cv2.WaitKey();
                    return;
                }

                Mat inputBlob = CvDnn.BlobFromImage(frame, 1 / 255d, new OpenCvSharp.Size(544, 544), new Scalar(), true, false);
                var results = darknet.Detect(frame, 0.5f);

                foreach (var result in results)
                {
                    var p1 = new OpenCvSharp.Point(result.x, result.y);
                    var p2 = new OpenCvSharp.Point(result.x + result.w, result.y + result.h);

                    if (CheckContoursAmount(contours, binary, p1, p2))
                    {
                        //Formatting labels
                        var label = $"{classNames[result.obj_id]}";

                        var textSize = Cv2.GetTextSize(label,
                            HersheyFonts.HersheyTriplex,
                            0.5,
                            1,
                            out var baseline);

                        Cv2.Rectangle(originalBackground, p1, p2, Colors[result.obj_id], -1);

                        Cv2.PutText(
                            originalBackground,
                            label,
                            new OpenCvSharp.Point((int)(result.x + (result.w / 2.0) - (textSize.Width / 2.0)), (int)(result.y + (result.h / 2.0))),
                            HersheyFonts.HersheyTriplex,
                            0.5,
                            Scalar.Black);
                    }  
                }

                bWorker.ReportProgress(0, originalBackground);
            }
        }

        private void OnPropertyChanged([CallerMemberName] string propertyName = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        private void LoadMap_Click(object sender, RoutedEventArgs e)
        {
            OpenCvWorker.RunWorkerAsync();
        }
    }
}
