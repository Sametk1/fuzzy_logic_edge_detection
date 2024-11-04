clc;
clear all;
close all;

% Görüntü Klasörünü Seç
images = uigetdir('F:\images ..','Görüntü Klasörünü Seç');
fnames = dir(images);
f = filesep;

% Ground Truth Klasörünü Seç
groundtruth = uigetdir('F:\groundtruth...','Ground Truth Klasörünü Seç');
names = dir(groundtruth);

A = {'.jpg','.bmp','.png','.tif'};
tic;

% Fuzzy Edge Detection FIS Oluşturma
sx = 0.1; % Daha yüksek sx ve sy daha düşük verimlilik
sy = 0.1;
edgeFIS = mamfis('Name','edgeDetection');
edgeFIS = addInput(edgeFIS,[-1 1],'Name','Ix');
edgeFIS = addInput(edgeFIS,[-1 1],'Name','Iy');
edgeFIS = addMF(edgeFIS,'Ix','gaussmf',[sx 0],'Name','zero');
edgeFIS = addMF(edgeFIS,'Iy','gaussmf',[sy 0],'Name','zero');
edgeFIS = addMF(edgeFIS,'Ix','gaussmf',[sx 0.5],'Name','medium');
edgeFIS = addMF(edgeFIS,'Iy','gaussmf',[sy 0.5],'Name','medium');
edgeFIS = addMF(edgeFIS,'Ix','gaussmf',[sx 1],'Name','large');
edgeFIS = addMF(edgeFIS,'Iy','gaussmf',[sy 1],'Name','large');
edgeFIS = addMF(edgeFIS,'Ix','gaussmf',[sx -0.5],'Name','small');
edgeFIS = addMF(edgeFIS,'Iy','gaussmf',[sy -0.5],'Name','small');
edgeFIS = addOutput(edgeFIS,[0 1],'Name','Iout');
wa = 0.1;
wb = 1;
wc = 1;
ba = 0;
bb = 0;
bc = 0.5;
edgeFIS = addMF(edgeFIS,'Iout','trimf',[wa wb wc],'Name','white');
edgeFIS = addMF(edgeFIS,'Iout','trimf',[ba bb bc],'Name','black');
r1 = "If Ix is zero and Iy is zero then Iout is white";
r2 = "If Ix is not zero or Iy is not zero then Iout is black";
r3 = "If Ix is large and Iy is small then Iout is black";
r4 = "If Ix is small and Iy is large then Iout is black";
r5 = "If Ix is large and Iy is zero then Iout is black";
r6 = "If Ix is zero and Iy is large then Iout is black";
r7 = "If Ix is medium and Iy is zero then Iout is black";
r8 = "If Ix is zero and Iy is medium then Iout is black";
edgeFIS = addRule(edgeFIS,[r1 r2 r3 r4 r5 r6 r7 r8]);

for k = 1:length(fnames)
    [pathstr,name,ext] = fileparts(fnames(k).name);
    if strcmpi(ext,'.jpg')==1
        rgb = imread([images,filesep,fnames(k).name]);
        if(size(rgb,3) > 1)
             f = rgb2gray(rgb); 
        else
             f = rgb;
        end
        
        sprintf('%d',k)
        t1 = imread([groundtruth,filesep,names(k).name]);
        sprintf('%d',k)   

        % Fuzzy Edge Detection
        Gx = [-1 1];
        Gy = Gx';
        Ix = conv2(double(f), Gx, 'same');
        Iy = conv2(double(f), Gy, 'same');
        Ix_scaled = 2 * (Ix - min(Ix(:))) / (max(Ix(:)) - min(Ix(:))) - 1;
        Iy_scaled = 2 * (Iy - min(Iy(:))) / (max(Iy(:)) - min(Iy(:))) - 1;
        Ieval = zeros(size(f));
        for ii = 1:size(f,1)
            Ieval(ii,:) = evalfis(edgeFIS,[Ix_scaled(ii,:); Iy_scaled(ii,:)]');
        end
        
        % Canny ve Sobel kenar algılama
        c = edge(f,'canny',0.3);
        b = edge(f,'sobel',0.1);
        
        % Precision ve Recall değerlerinin hesaplanması
        [precision_c, recall_c, f_measure_c, pr_c] = calculatePRFMeasure(t1, c);
        [precision_b, recall_b, f_measure_b, pr_b] = calculatePRFMeasure(t1, b);
        [precision_f, recall_f, f_measure_f, pr_f] = calculatePRFMeasure(t1, Ieval);
        % Tersine çevrilen fuzzy edge detection çıktısını eşikle
Ieval_complement = imcomplement(Ieval); % Tersine çevrilen fuzzy edge detection çıktısı
Ieval_complement_thresholded = imbinarize(Ieval_complement, 0.5); % Gri alanları beyaz yap
Ieval_complement_thresholded = uint8(Ieval_complement_thresholded) * 255; % Çıktıyı 0-255 aralığına dönüştür
Ieval_complement_thresholded = imcomplement(Ieval_complement_thresholded); % Tersine çevrilen çıktıyı geri çevirerek arka planı beyaz, çizgileri siyah yap

% Görselleştirme
figure,
subplot(2,2,4), imshow(Ieval_complement_thresholded), colormap('gray'), title(['Fuzzy Edge Detection, PR: ', num2str(pr_f), ', F-measure: ', num2str(f_measure_f)]);
subplot(2,2,3), imshow(c), title(['Canny, PR: ', num2str(pr_c), ', F-measure: ', num2str(f_measure_c)]);
subplot(2,2,1), imshow(rgb), title('BSD Image');
subplot(2,2,2), imshow(t1), title('Ground Truth');
    end
end

function [precision, recall, f_measure, pr] = calculatePRFMeasure(ground_truth, edge_result)
    tp = sum(ground_truth(:) & edge_result(:));
    fp = sum(~ground_truth(:) & edge_result(:));
    fn = sum(ground_truth(:) & ~edge_result(:));

    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    
    if isnan(precision) || isnan(recall) || (precision + recall == 0)
        f_measure = NaN;
    else
        f_measure = 2 * (precision * recall) / (precision + recall);
    end
    
    pr = tp / (tp + fp + fn);
end