% Import the image
Irgb = imread('C:\Users\pc\Desktop\basic.png');

Igray = rgb2gray(Irgb);

% Convert Irgb to grayscale so that you can work with a 2-D array 
% instead of a 3-D array. To do so, use the rgb2gray function.
figure
image(Igray,'CDataMapping','scaled')
colormap('gray')
title('Input Image in Grayscale')

% The evalfis function for evaluating fuzzy inference systems supports 
% only single-precision and double-precision data.
% Therefore, convert Igray to a double array using the im2double function.
I = im2double(Igray);

Gx = [-1 1];
Gy = Gx';
Ix = conv2(I,Gx,'same');
Iy = conv2(I,Gy,'same');

% Plot the image gradients.

figure
image(Ix,'CDataMapping','scaled')
colormap('gray')
title('Ix')

figure
image(Iy,'CDataMapping','scaled')
colormap('gray')
title('Iy')

% Create a fuzzy inference system (FIS) for edge detection, edgeFIS.

edgeFIS = mamfis('Name','edgeDetection');

% Specify the image gradients, Ix and Iy, as the inputs of edgeFIS

edgeFIS = addInput(edgeFIS,[-1 1],'Name','Ix');
edgeFIS = addInput(edgeFIS,[-1 1],'Name','Iy');

% Specify a zero-mean Gaussian membership function for each input. 
% If the gradient value for a pixel is 0, 
% then it belongs to the zero membership function with a degree of 1.
sx = 0.1; %higher sx and sy lower efficiency
sy = 0.1;
edgeFIS = addMF(edgeFIS,'Ix','gaussmf',[sx 0],'Name','zero');
edgeFIS = addMF(edgeFIS,'Iy','gaussmf',[sy 0],'Name','zero');

% Add additional membership functions for Ix and Iy
edgeFIS = addMF(edgeFIS,'Ix','gaussmf',[sx 0.5],'Name','medium');
edgeFIS = addMF(edgeFIS,'Iy','gaussmf',[sy 0.5],'Name','medium');
edgeFIS = addMF(edgeFIS,'Ix','gaussmf',[sx 1],'Name','large');
edgeFIS = addMF(edgeFIS,'Iy','gaussmf',[sy 1],'Name','large');
edgeFIS = addMF(edgeFIS,'Ix','gaussmf',[sx -0.5],'Name','small');
edgeFIS = addMF(edgeFIS,'Iy','gaussmf',[sy -0.5],'Name','small');

% Specify the intensity of the edge-detected image as an output of edgeFIS.
edgeFIS = addOutput(edgeFIS,[0 1],'Name','Iout');

% Specify the triangular membership functions, white and black, for Iout.
wa = 0.2;
wb = 1;
wc = 1;
ba = 0;
bb = 0;
bc = 0.5;
edgeFIS = addMF(edgeFIS,'Iout','trimf',[wa wb wc],'Name','white');
edgeFIS = addMF(edgeFIS,'Iout','trimf',[ba bb bc],'Name','black');

% Plot the membership functions of the inputs and outputs of edgeFIS.
figure
subplot(2,2,1)
plotmf(edgeFIS,'input',1)
title('Ix')
subplot(2,2,2)
plotmf(edgeFIS,'input',2)
title('Iy')
subplot(2,2,[3 4])
plotmf(edgeFIS,'output',1)
title('Iout')

% Specify FIS Rules

r1 = "If Ix is zero and Iy is zero then Iout is white";
r2 = "If Ix is not zero or Iy is not zero then Iout is black";
r3 = "If Ix is large and Iy is small then Iout is black";
r4 = "If Ix is small and Iy is large then Iout is black";
r5 = "If Ix is large and Iy is zero then Iout is black";
r6 = "If Ix is zero and Iy is large then Iout is black";
r7 = "If Ix is medium and Iy is zero then Iout is black";
r8 = "If Ix is zero and Iy is medium then Iout is black";

% Add the rules to the FIS
edgeFIS = addRule(edgeFIS,[r1 r2 r3 r4 r5 r6 r7 r8]);
edgeFIS.Rules

% Evaluate FIS

Ieval = zeros(size(I));
for ii = 1:size(I,1)
    Ieval(ii,:) = evalfis(edgeFIS,[(Ix(ii,:));(Iy(ii,:))]');
end

% Plot results 
figure
image(I,'CDataMapping','scaled')
colormap('gray')
title('Original Grayscale Image')

% Plot detected edges
figure
image(Ieval,'CDataMapping','scaled')
colormap('gray')
title('Edge Detection Using Fuzzy Logic')
