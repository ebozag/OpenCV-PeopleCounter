/************************************************************************************************************* 

LABORATORIO: DETECCION DE MOVIMIENTO - BACKGROUND SUBTRACTION

Este laboratorio tiene como objetivo usar las funciones de OpenCV para realizar detecci�n de movimiento. 
Las funciones que se usan son las de: Background Subtraction. Entre estas funciones las m�s conocidas son 
las siguientes:
BackgroundSubtractorMOG2

NOTA: usar como dato de entrada una secuencia de video que permita visualizar la detecci�n de los objetos 
en movimiento. 

*********************************************************************************************************************/

#include <stdio.h>
#include <iostream>
#include "opencv2/highgui.hpp"
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>


using namespace cv;
using namespace std;

// VARIABLES GLOBALES
//Primero, separamos espacio de memoria para almacenar el frame actual, y las m�scaras de foreground que se obtendr�n con los diferentes algoritmos de BS
Mat frame_actual; //frame actual
Mat mascaraMOG2; //mascara de foreground generada por MOG2
Ptr<BackgroundSubtractor> substractorMOG2; //sustractor de background MOG2
int keyboard; //input 
Point point1, point2; /* puntos de la linea de referencia */
int drag = 0;                                                                                                                                                           
int select_flag = 0;

// Función para dibujar la línea con el ratón
void mouseHandler(int event, int x, int y, int flags, void* param)
{
    if (event == CV_EVENT_LBUTTONDOWN && !drag)
    {
        /* left button clicked. ROI selection begins */
        point1 = Point(x, y);
        drag = 1;
    }
    
    if (event == CV_EVENT_MOUSEMOVE && drag)
    {
        /* mouse dragged. ROI being selected */
        Mat img1 = frame_actual.clone();
        point2 = Point(x, y);
        line(img1, point1, point2, CV_RGB(255, 0, 0), 2, 8, 0);
        imshow("Frame", img1);
    }
    
    if (event == CV_EVENT_LBUTTONUP && drag)
    {
        point2 = Point(x, y);
        drag = 0;
    }
    
    if (event == CV_EVENT_LBUTTONUP)
    {
       /* ROI selected */
        select_flag = 1;
        drag = 0;
    }
}

// Función para verificar si dos líneas se intersectan,
// basado en:
// http://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
bool ccw(Point2f A,Point2f B,Point2f C){
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x);
}

//Return true if line segments AB and CD intersect
bool intersecta(Point2f A,Point2f B,Point2f C,Point2f D){
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D);
}


//Función principal
int main(int argc, char* argv[])
{
    char nombre_video[50];

    //Iniacializa los "substractores de background"	
    substractorMOG2 = createBackgroundSubtractorMOG2(); //MOG2  

    //Carga el video en la variable capture
    VideoCapture capture("Video_003.avi");

    //Flag en caso de haber un error al abrir el video
    if (!capture.isOpened()){
        printf("No fue posible abrir el archivo de video");
        getchar();
        exit(EXIT_FAILURE);
    }

    //Toma la primera imagen y permite dibujar la línea para realizar el conteo
    //Capturamos el primer frame
    if (!capture.read(frame_actual)) {
        printf("No fue posible leer el frame");
        getchar();
        exit(EXIT_FAILURE);
    }

    printf("Dibuje la línea para el conteo y presione la barra espaciadora para continuar.");
    fflush(stdout); 
    while ((char)keyboard != 32){
        cvSetMouseCallback("Frame", mouseHandler, NULL);
        Mat img1 = frame_actual.clone();
        line(img1, point1, point2, CV_RGB(255, 0, 0), 2, 8, 0);
        imshow("Frame", img1);
        keyboard = waitKey(30);
    }
    
    // Variables para detección
    Mat im_detecciones;
    vector<cv::KeyPoint> personas;
    vector<cv::Point2f> prevPersonas;
    Point2f persona;
    int i = 0;  //Contador para recorrer las personas detectadas
    int contador = 0; // Contador de cruces por la línea

    // Preparar los parámetros para el detector
    cv::SimpleBlobDetector::Params params;
    params.minDistBetweenBlobs = 1.0f;
    params.filterByInertia = false;
    params.filterByConvexity = false;
    params.filterByColor = false;
    params.filterByCircularity = false;
    params.filterByArea = true;
    params.minArea = 100.0f;
    params.maxArea = 2500.0f;

    // Preparar el detector con los parámetros definidos
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
    
    //Flag para acabar el programa: mientras que no se presione tecla ESC
    while ((char)keyboard != 27){
        //Capturamos el primer frame, si hubo error acaba la ejecuci�n
        if (!capture.read(frame_actual)) {
            printf("No fue posible leer el frame");
            getchar();
            exit(EXIT_FAILURE);
        }

        // cada frame es usado para calcular la m�scara de foreground y para actualizar el background
        substractorMOG2->apply(frame_actual, mascaraMOG2);	// aplico el algoritmo MOG2 y obtengo la m�scara de foreground

        // Tratamiento de la máscara de foreground, para eliminar pequeños objetos 
        // y llenar pequeños huecos. (Closing y opening))
        dilate(mascaraMOG2, mascaraMOG2, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)) ); 
        erode(mascaraMOG2, mascaraMOG2, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)) );   
        erode(mascaraMOG2, mascaraMOG2, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)) );
        dilate(mascaraMOG2, mascaraMOG2, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)) ); 

        //muestra el frame actual y su máscara
        line(frame_actual, point1, point2, CV_RGB(255, 0, 0), 2, 8, 0);
        imshow("Frame", frame_actual);
        //imshow("Mascara Foreground MOG2", mascaraMOG2);
        imshow("Mascara Foreground MOG2 Mejorada", mascaraMOG2);

        // Detección de BLOBs
        // https://www.learnopencv.com/blob-detection-using-opencv-python-c/ 
        cv::KeyPoint::convert(personas,prevPersonas);
        detector->detect(mascaraMOG2, personas);                
        drawKeypoints(frame_actual, personas, im_detecciones, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

        // Detección de desplazamiento de personas
        if(prevPersonas.size() >= personas.size()){
            i=0;
            for(std::vector<cv::KeyPoint>::iterator blobIterator = personas.begin(); blobIterator != personas.end(); blobIterator++){
                persona = Point2f(blobIterator->pt.x, blobIterator->pt.y);
                // Determino la longitud y no tomo en cuenta aquellas muy grandes.
                double longitud = cv::norm(persona-prevPersonas[i]);
                if (longitud < 10){
                    line(im_detecciones, persona, prevPersonas[i], CV_RGB(0, 255, 0), 1, 8, 0);
                    // Contar los cruces de línea
                    if (intersecta(persona,prevPersonas[i],point1,point2)){
                        contador++;
                        printf("%d\n",contador);
                    }
                }
                i++;
            }
        }
        // Mostrar personas y desplazamiento
        imshow("keypoints", im_detecciones );
        //get the input from the keyboard
        keyboard = waitKey(30);
    }

    //delete capture object
    capture.release();

    //destroy GUI windows
    destroyAllWindows();
    return EXIT_SUCCESS;
}
