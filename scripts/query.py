import pandas as pd
import os

def get_imagenette():
    df = pd.read_csv(os.path.join('data', 'fusion_output_imagenette.csv'))
    return df

def get_intelimage():
    df = pd.read_csv(os.path.join('data', 'fusion_output_intel_image.csv'))
    return df

def main():
    

    """
    \begin{itemize}
        \item \textbf{RQ1}: Which is the best extractor model for dataset Intel Image and Imagenette, using gradcam as saliency method, ResNet as classifier and concept average saliency (cas) as concept presence measure?
        \item \textbf{RQ2}: Which is the best saliency method for dataset Intel Image and Imagenette, using VGG as classifier?
        \item \textbf{RQ3}: For class "English springer" of dataset Imagenette, which is the best extractor of concepts?
        \item \textbf{RQ4}: Which classifier produce the best alignment, using florence or groundingdino as extractor method, Grad-cam as saliency method, and IntersectionOverUnion as concept presence measure for class "Street" in Intel Image dataset? 
    \end{itemize}
    """

    #RQ1
    df_imagenette = get_imagenette() #COLUMNS: Saliency,Classifier,Extractor,Concept Presence,Classes,Fin,Head,Ears,Muzzle,Paws,Tail,Body,Button,Speaker,Digital display,Chain bar,Rose window,Facade,Bell Tower,Bell,Cab,Wheel,Headlights,Hose,Tank,Nozzle,Golf ball dimples,Logo,Canopy,WOE Score
    df_intelimage = get_intelimage() 
    
    print('################### RQ1 ###################')
    df_imagenette = df_imagenette[(df_imagenette['Classifier'] == 'resnet') & (df_imagenette['Saliency'] == 'gradcam') & (df_imagenette['Concept Presence'] == 'cas') ]
    df_imagenette = df_imagenette.groupby('Extractor')['WOE Score'].mean().reset_index()
    print('Imagenette')
    print(df_imagenette)
    
    df_intelimage = df_intelimage[(df_intelimage['Classifier'] == 'resnet') & (df_intelimage['Saliency'] == 'gradcam') & (df_intelimage['Concept Presence'] == 'cas') ]
    df_intelimage = df_intelimage.groupby('Extractor')['WOE Score'].mean().reset_index()
    print('\nIntel Image')
    print(df_intelimage)

    #RQ2
    print('\n################### RQ2 ###################')
    df_imagenette = get_imagenette() #COLUMNS: Saliency,Classifier,Extractor,Concept Presence,Classes,Fin,Head,Ears,Muzzle,Paws,Tail,Body,Button,Speaker,Digital display,Chain bar,Rose window,Facade,Bell Tower,Bell,Cab,Wheel,Headlights,Hose,Tank,Nozzle,Golf ball dimples,Logo,Canopy,WOE Score
    df_intelimage = get_intelimage()

    df_imagenette = df_imagenette[(df_imagenette['Classifier'] == 'vgg') ]
    df_imagenette = df_imagenette.groupby('Saliency')['WOE Score'].mean().reset_index()
    print('Imagenette')
    print(df_imagenette)

    df_intelimage = df_intelimage[(df_intelimage['Classifier'] == 'vgg') ]
    df_intelimage = df_intelimage.groupby('Saliency')['WOE Score'].mean().reset_index()
    print('\nIntel Image')
    print(df_intelimage)

    #RQ3
    print('\n################### RQ3 ###################')
    df_imagenette = get_imagenette() #COLUMNS: Saliency,Classifier,Extractor,Concept Presence,Classes,Fin,Head,Ears,Muzzle,Paws,Tail,Body,Button,Speaker,Digital display,Chain bar,Rose window,Facade,Bell Tower,Bell,Cab,Wheel,Headlights,Hose,Tank,Nozzle,Golf ball dimples,Logo,Canopy,WOE Score
    df_imagenette = df_imagenette[(df_imagenette['Classes'] == 'English springer') ]
    df_imagenette = df_imagenette.groupby('Extractor')['WOE Score'].mean().reset_index()
    print('Imagenette')
    print(df_imagenette)

    #RQ4
    print('\n################### RQ4 ###################')
    df_intelimage = get_intelimage()
    df_intelimage = df_intelimage[(df_intelimage['Extractor'] == 'florence') & (df_intelimage['Saliency'] == 'gradcam') & (df_intelimage['Concept Presence'] == 'iou') & (df_intelimage['Classes'] == 'street') ]
    df_intelimage = df_intelimage.groupby('Classifier')['WOE Score'].mean().reset_index()
    print('Intel Image', 'florence')
    print(df_intelimage)

    df_intelimage = get_intelimage()
    df_intelimage = df_intelimage[(df_intelimage['Extractor'] == 'groundingdino') & (df_intelimage['Saliency'] == 'gradcam') & (df_intelimage['Concept Presence'] == 'iou') & (df_intelimage['Classes'] == 'street') ]
    df_intelimage = df_intelimage.groupby('Classifier')['WOE Score'].mean().reset_index()
    print('Intel Image', 'groundingdino')
    print(df_intelimage)









if __name__ == '__main__':
    main()