from PIL import Image
import numpy as np
import torch

# colour map
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    mask = mask.data.cpu().numpy()
    n, h, w = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images): 
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_,j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs

def decode_heatmaps(heatmaps, num_images=1): 
    
    print(list(heatmaps.size())) 
    if isinstance(heatmaps, list):
        preds_list = []
        for pred in heatmaps:
            preds_list.append(pred[-1].data.cpu().numpy())
        heatmaps = np.concatenate(preds_list, axis=0)
    else:
        heatmaps = heatmaps.data.cpu().numpy()
    
    print( heatmaps.shape)     
    n, c, h, w = heatmaps.shape 
     
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, 2, h, w, 3), dtype=np.uint8)
    for i in range(num_images): 
        img = Image.new('RGB', (w, h))
        pixels = img.load() 
        # R_Knee
        R_Knee = heatmaps[i,1, :, :]
        print('-max----min---knee-')
        print(np.max(R_Knee))
        print(np.min(R_Knee))
        
        R_Knee[R_Knee<0] = 0
        #if(np.max(R_Knee) != 0): 
        #    R_Knee = R_Knee/(np.max(R_Knee))
         
        print(np.max(R_Knee))
        R_Knee = (R_Knee*255.0).astype(np.uint8)
        print(np.max(R_Knee))
         
        for j_, j in enumerate(R_Knee):  
            for k_, k in enumerate(j):
                    pixels[k_,j_] = (k,k,k)  
        outputs[i, 0] = np.array(img)
        
        # R_Shoulder
        R_Shoulder = heatmaps[i,12, :, :] 
        
        R_Shoulder[R_Shoulder<0] = 0 
        #if(np.max(R_Shoulder) != 0): 
        #   R_Shoulder = R_Shoulder/(np.max(R_Shoulder)) 
        R_Shoulder = (R_Shoulder*255.0).astype(np.uint8)
        for j_, j in enumerate(R_Shoulder):  
            for k_, k in enumerate(j):
                    pixels[k_,j_] = (k,k,k)  
        outputs[i, 1] = np.array(img)  
     
    return outputs
 

def decode_predictions(preds, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    if isinstance(preds, list):
        preds_list = []
        for pred in preds:
            preds_list.append(pred[-1].data.cpu().numpy())
        preds = np.concatenate(preds_list, axis=0)
    else:
        preds = preds.data.cpu().numpy()

    preds = np.argmax(preds, axis=1)
    n, h, w = preds.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(preds[i, 0]), len(preds[i])))
      pixels = img.load()
      for j_, j in enumerate(preds[i, :, :]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def inv_preprocess(imgs, num_images, img_mean):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    imgs = imgs.data.cpu().numpy()
    n, c, h, w = imgs.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (np.transpose(imgs[i], (1,2,0)) + img_mean).astype(np.uint8)
    return outputs
