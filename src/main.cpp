/*//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Universidade Federal de Itajubá - campus Itabira
Engenharia de Computação - Computação Gráfica e Processamento Digital de Imagens

Projeto: Separação de Grãos de Café.

Integrantes:
    Diognei de Matos          R.A. 28484
    Felipe Marinho Tavares    R.A. 27305
    João Guilherme Costa      R.A. 28631
    Yuri Souza                R.A. 24543

  Prof. Dr. Giovani Bernardes


Este código foi desenvolvido como Projeto#2 da disciplina de ECO034 da Unifei - campus Itabira.

Seu objetivo é realizar a identificação e análise de grãos de café utilizando processamento digital de sinais e visão computacional. 

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;

//=========================================================_Configuraçao da execução por variaveis globais
bool const calibracao = false;
bool const variasImagens = true;  

int startX=104, startY=173, width=479, height=351; //Coordenadas do corte padrão da imagem para identificação

//---------_Adaptar de acordo com o diretório onde as imagens estão na máquina
String imagemCalibracao = "imagemCalibracao.jpg"; //Imagem de calibração desejada (calibracao == true)
String pathImage = "Filtered example.jpg";  //Imagem da única image que se deseja análisar (variasImagens == false)
cv::String path("/home/chiruno/Desktop/Coffee Final Version/separacaograosdecafe/images/*.jpg"); //(variasImagens == true)
String folderName = "Identified"; //Nome da pasta que serão salvas as várias imagens processadas. (variasImagens == true)

//==========================================================================================================

//  Criaram-se classes de café para caso em futuros projetos elas armazenem mais informações
//como de deslocalmento, além de threshold de HSV.
//  Os valores devem ser preenchidos na variável de acordo com a análise empirica na fase de calibração.
class CafeVerde {
  public:
    int lowH, highH, lowS, highS, lowV, highV;
    string name;

    CafeVerde(){
      name = "Verde";
      lowH = 37;
      highH = 179;
      lowS = 111;
      highS = 255;
      lowV = 0;
      highV = 39;
    }
};
class CafeAmarelo {
  public:
    int lowH, highH, lowS, highS, lowV, highV;
    string name;

    CafeAmarelo(){
      name = "Amarelo";
      lowH = 22;
      highH = 28;
      lowS = 122;
      highS = 255;
      lowV = 44;
      highV = 255;
    }
};
class CafeLaranja {
  public:
    int lowH, highH, lowS, highS, lowV, highV;
    string name;

    CafeLaranja(){
      name = "Laranja";
      lowH = 5;
      highH = 32;
      lowS = 173;
      highS = 255;
      lowV = 27;
      highV = 52;
    }
};
class CafePreto {
  public:
    int lowH, highH, lowS, highS, lowV, highV;
    string name;

    CafePreto(){
      name = "Preto";
      lowH = 0;
      highH = 22;
      lowS = 0;
      highS = 255;
      lowV = 0;
      highV = 27;
    }
};

//-----------------------------------------------------Função para execução da interface de calibragem e ajuste de valores de threshold. 
//    *Lembrar de preencher os valores desejados de threshold nas classes de café. 
void interfaceCalibracao()
{
  namedWindow("Control", CV_WINDOW_AUTOSIZE); // Criação da janela de controle

  int iLowH = 0;
  int iHighH = 179;

  int iLowS = 0;
  int iHighS = 255;

  int iLowV = 0;
  int iHighV = 255;

  //Criação de trackbars na janela de controle
  cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
  cvCreateTrackbar("HighH", "Control", &iHighH, 179);

  cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
  cvCreateTrackbar("HighS", "Control", &iHighS, 255);

  cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
  cvCreateTrackbar("HighV", "Control", &iHighV, 255);

  while (true){
    Mat imgOriginal = imread(imagemCalibracao);

    Mat imgHSV;
    cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Conversão para HSV

    Mat imgThresholded;
    inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold da imagem

    //Abertura - Retira pequenos objetos do fundo
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

    //Fechamento - Preenche pequenos buracos no fundo
    dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

    imshow("Thresholded Image", imgThresholded); //Mostra a imagem tratada
    imshow("Original", imgOriginal); //Mostra a imagem original

    if (waitKey(30) == 27) //Espera esc por 30ms. Se precionado, para o loop.
    {
      cout << "Esc foi precionado pelo usuário\n";
      break;
    }
  }
}

//-----------------------------------------------------------------------------------Função que realiza o processamento na dada imagem
void classificacafe(Mat &destiny)
{
  //Aplicação do Filtro pyrMeanShift para constrate dos grãos com backgroud e suavisão da imagem
  Mat shift;
  pyrMeanShiftFiltering(destiny, shift, 21, 51);
  if(!variasImagens)
    imwrite("shift.png", shift);

  //Transformando a imagem para binário (255 ou 0)
  Mat gray;
  cvtColor(shift, gray, CV_BGR2GRAY);
  if(!variasImagens)
    imwrite("gray.png", gray);

  Mat thresh;
  threshold(gray, thresh, 0,255, THRESH_BINARY_INV | THRESH_OTSU);
  if(!variasImagens)
    imwrite("thresh.png", thresh);

  //Removendo ruído e simplificando a imagem com dois openings
  Mat kernel = Mat(3, 3, CV_8UC1, Scalar(1));

  Mat opening;
  morphologyEx(thresh, opening, MORPH_OPEN, kernel, Point(-1,-1), 1);
  if(!variasImagens)
    imwrite("opening1.png", opening);

  morphologyEx(opening, opening, MORPH_OPEN, kernel, Point(-1,-1), 2);
  if(!variasImagens)
    imwrite("opening2.png", opening);

  //------------------------------------------Rotulação dos grãos da imagem por uso do algoritmo de Componentes Conectados.
  Mat stats,centroids;
  Mat labelImage(destiny.size(),CV_32S);
  int nLabels = connectedComponentsWithStats(opening,labelImage,stats,centroids,8);


  //---------- Caso se deseje dados detalhados de cada componente conexo identificado
  /*
  cout << "Number of connected components = " << nLabels << endl << endl;

  cout << "Show statistics and centroids:" << endl;
  cout << "stats:" << endl << "(left,top,width,height,area)" << endl << stats << endl << endl;
  cout << "centroids:" << endl << "(x, y)" << endl << centroids << endl << endl;

  cout << "Component 1 stats:" << endl;
  cout << "CC_STAT_LEFT   = " << stats.at<int>(1,CC_STAT_LEFT) << endl;
  cout << "CC_STAT_TOP    = " << stats.at<int>(1,CC_STAT_TOP) << endl;
  cout << "CC_STAT_WIDTH  = " << stats.at<int>(1,CC_STAT_WIDTH) << endl;
  cout << "CC_STAT_HEIGHT = " << stats.at<int>(1,CC_STAT_HEIGHT) << endl;
  cout << "CC_STAT_AREA   = " << stats.at<int>(1,CC_STAT_AREA) << endl;
  */


  //----------- Caso se deseje imprimir uma imagem colorida representado os individuos identificados
  //*
  vector<Vec3b> colors(nLabels);
  colors[0] = Vec3b(0,0,0);
  for (int label =1; label<nLabels; ++label)
    colors[label]  = Vec3b((rand()&255), (rand()&255), (rand()&255) );
  
  Mat dst_final(destiny.size(),CV_8UC3);
  for(int r = 0; r < dst_final.rows; ++r)
    for(int c = 0; c < dst_final.cols; ++c){
        int label = labelImage.at<int>(r, c);
        Vec3b &pixel = dst_final.at<Vec3b>(r, c);
        pixel = colors[label];
     }
  if(!variasImagens){
    imshow("connectedComponents", dst_final);
    imwrite("connectedComponents.png", dst_final);
  }
  //*/
  

  //-----------------------------------------------------------------------------------------Fazendo a Mat de cada classificação
  CafeVerde *cv = new CafeVerde();
  CafeAmarelo *ca = new CafeAmarelo();
  CafeLaranja *cl = new CafeLaranja();
  CafePreto *cp = new CafePreto();

  Mat hsvThVerde, hsvThAmarelo, hsvThLaranja, hsvThPreto;

  Mat imageHSV;
  //Convertendo as cores da imagem original para HSV
  cvtColor(destiny, imageHSV, COLOR_BGR2HSV);
  //Criando um threshold HSV para cada classificação de café
  inRange(imageHSV, Scalar(cv->lowH, cv->lowS, cv->lowV), Scalar(cv->highH, cv->highS, cv->highV), hsvThVerde);
  inRange(imageHSV, Scalar(ca->lowH, ca->lowS, ca->lowV), Scalar(ca->highH, ca->highS, ca->highV), hsvThAmarelo);
  inRange(imageHSV, Scalar(cl->lowH, cl->lowS, cl->lowV), Scalar(cl->highH, cl->highS, cl->highV), hsvThLaranja);
  inRange(imageHSV, Scalar(cp->lowH, cp->lowS, cp->lowV), Scalar(cp->highH, cp->highS, cp->highV), hsvThPreto);

  if(!variasImagens){
    imwrite("CafeVerde.png", hsvThVerde);
    imwrite("CafeAmarelo.png", hsvThAmarelo);
    imwrite("CafeLaranja.png", hsvThLaranja);
    imwrite("CafePreto.png", hsvThPreto);
  }

//----------------------------------------------------------Inicio da Classificação de cada grão
  for (int j = 1; j < nLabels; j++) {
    int area = stats.at<int>(j, CC_STAT_AREA);
    int left = stats.at<int>(j, CC_STAT_LEFT);
    int top  = stats.at<int>(j, CC_STAT_TOP);
    int width = stats.at<int>(j, CC_STAT_WIDTH);
    int height  = stats.at<int>(j, CC_STAT_HEIGHT);

    //Definição do Triangulo respectivo ao grão.
    rectangle( destiny, Point(left,top), Point((left+width),(top+height)), Scalar(0,0,255),1 );

    //Integral dos grãos referente aos tipos de classificação diferentes.
    int contVerde = 0, contPreto = 0, contLaranja = 0, contAmarelo = 0;

    Mat auxVerde(hsvThVerde, Rect(left, top, width, height));
    contVerde = countNonZero(auxVerde);

    Mat auxPreto(hsvThPreto, Rect(left, top, width, height));
    contPreto = countNonZero(auxPreto);

    Mat auxAmarelo(hsvThAmarelo, Rect(left, top, width, height));
    contAmarelo = countNonZero(auxAmarelo);

    Mat auxLaranja(hsvThLaranja, Rect(left, top, width, height));
    contLaranja = countNonZero(auxLaranja);

    //----------------------------------------------Impressão dos grãos e a quantidade de pixeis para cada classificação
    if(!variasImagens){
      cout << "Grão: "<< j << endl;
      cout << "Verde "<< contVerde <<  endl;
      cout << "Amarelo "<< contAmarelo << endl;
      cout << "Laranja "<< contLaranja << endl;
      cout << "Preto "<< contPreto << endl;
      cout << endl;
    }

    //------------------------------------------------------Impressão na imagem original do grão identificado
    //  Aqui é definida a classificação de fato do grão de acordo com o maior número de pixeis identificaveis
    //pelo conjunto de Thresholds em HSV.
    if(contVerde > contAmarelo && contVerde > contLaranja && contVerde > contPreto)
      putText(destiny, to_string(j)+"v", Point(left+20,top+20), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,0,0), 2); 
    else if(contAmarelo > contVerde && contAmarelo > contLaranja && contAmarelo > contPreto)
      putText(destiny, to_string(j)+"a", Point(left+20,top+20), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,0,0), 2); 
    else if(contLaranja > contVerde && contLaranja > contAmarelo && contLaranja > contPreto)
      putText(destiny, to_string(j)+"l", Point(left+20,top+20), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,0,0), 2); 
    else if(contPreto > contAmarelo && contPreto > contLaranja && contPreto > contVerde)
      putText(destiny, to_string(j)+"p", Point(left+20,top+20), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,0,0), 2); 
    else
      putText(destiny, to_string(j)+"*", Point(left+20,top+20), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,0,0), 2); 

    if(!variasImagens){
      imshow( "Resultado", destiny);
      imwrite( "Resultado.png", destiny);
    }
  }
}


//----------------------------------------_Função main - Seu papel é inicializar o processamento de acordo com a config. selecionada.
int main( int argc, char** argv )
{
  if(calibracao){
    printf("\n::Interface de calibração inicializada. Para sair aperte ESC.\n");
    interfaceCalibracao();
  }
  else{
    printf("Algoritmo de seleção inicializado...\n");
    Mat source, destiny;
    if(!variasImagens){
      printf("Processamento de identificação de uma imagem inicializado...\n");
      source = imread(pathImage); //Imagem que será analisada no momento.
      if( !source.data ){ 
        printf("::Nenhum arquivo de entrada... Terminando execução. \n");
        return -1; 
      }

      Mat ROI(source, Rect(startX,startY,width,height));  //Recorte padrão da imagem para a parte de interesse

      ROI.copyTo(destiny);
      classificacafe(destiny);  //Chamada da função que processará a imagem e retornará os resultados.
      printf("Processamento finalizado. Pressione qualquer tecla para sair.\n");
      waitKey(0);
    }
    else{
      printf("Processamento de identificação de uma várias imagens inicializado...\n");
      int ct = 0; //Contador de imagens
      stringstream ss;

      String name = folderName+"_";  //Nome da pasta a ser criada com os arquivos.
      String type = ".jpg";

      String folderCreateCommand = "mkdir " + folderName;

      system(folderCreateCommand.c_str());
      
      vector<cv::String> fn;
      vector<cv::Mat> data;
      cv::glob(path,fn,true); //Recursivo

      for (size_t k=0; k<fn.size(); ++k){ //Loop para percorer as imagens da pasta desejada.
         cout << fn[k] << endl;
         source = cv::imread(fn[k]);
         if (source.empty()) 
          continue; 
         Mat ROI(source, Rect(startX,startY,width,height));
         ROI.copyTo(destiny);

        classificacafe(destiny);  //Chamada da função que processará a imagem e retornará os resultados.

        ss<<folderName<<"/"<<name<<(ct)<<type;

        String fullPath = ss.str();
        ss.str("");
        imwrite(fullPath, destiny);

        ct++;

        data.push_back(source);
      }
      printf("Processamento finalizado. Imagens salvas em: %s.\n", folderName.c_str());
    }
  }
  return 0;
}
