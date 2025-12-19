% Initial Version of DOA Estimation Using Hybrid CNN-AB-GRU and NCA
clc
clear all
close all
warning off
%%%%%%%%%%%%%% A-set parameters %%%%%%%%%%%%%
% [K d SNR]: Scenario1:[200 0.5 0] ;Scenario2:[500 0.5 0]; Scenario3:[500 0.7 0]
N=10;                                    %number of array element
K=500;                                   %number of data snapshot (signal length)
d=0.5;                                   %distance between elements inwavelengths

p_noise=1;                               %power of nois

DOAs=-30:14;
snr1=0;
for i=1:length(DOAs)
    DOAs_deg=[DOAs(i) DOAs(i)+8 DOAs(i)+16];
    DOAs_rad=DOAs_deg*pi/180;                      %doa's of signal in radian
    r=length(DOAs_rad);                            %number of DOAs = number of transmiters
    p_sig=ones(1,1);
    p_noise=p_sig/(10^(snr1/10));%power of incoming signals
    %%%%%%%%%%%%%% B-system modeling %%%%%%%%%%%%%
    %steering vector matrix. columns will containe the steering vectors of signals
    A=exp(-1i*2*pi*d*(0:N-1)'*sin([DOAs_rad(:).']));        %8-3 fourmola - page 22
    %signal and noise generationn
    sig=round(rand(r,K))*2-1;                           %grnerate random  bpsk symbols for each of the r signal
    noise=sqrt(p_noise)*(randn(N,K));                   %uncorolated noise
    X=A*diag(sqrt(p_sig))*sig+noise;                    %generating data matrix (X=As+n 12-3fourmola-page:23)

    %%%%%%%%%%%%  C-cov matrix and eigenvalues  %%%%%%%%%%%%%

    R=X*X'/K;                                           %spatial covariance matrix
    %%%% select uper triangel of R and put in y
    UR = triu(R,1);
    y0(i,:)=UR(:);
    yy=y0(i,:);
    ind=find(yy==0);
    yy(ind)=[];
    y(i,:)=yy;


    %%%%%%%%%%%%%%%%  D-estimate with Music
    nsig=r;
    covmat=R;
    est_DOAs_m = -musicdoa(covmat,nsig);
    est_DOAs_matlab(i,:)=sort(est_DOAs_m);


end


t(1,:)=DOAs;
t(2,:)=t(1,:)+8;
t(3,:)=t(1,:)+16;

mont_carlo=1;
%%%%%%%%%%%%%%%%  E-estimate with DNN
out_mont=zeros(size(t));
F=[(real(y))';(imag(y))'];
num_target=size(t,1);
tt_SVR=t;
tt=t';
F=F';
for q=1:num_target
    t=tt(:,q);
    p=F;
    ClassNum=length(t);
    p=rescale(p,0,1);
    [number_of_data,num_feature]=size(p);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%% CNN Network %%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    numFeatures = num_feature;
    numClasses = numel(unique(t));
    numHiddenNeuron = 20;

    layers = [
        sequenceInputLayer(numFeatures)
        convolution1dLayer(1,3)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(numHiddenNeuron)
        sigmoidLayer('Name','sig')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer('Name','classification')];


    %%%%%%%%%%%%%%%%%%%%%%%%%
    miniBatchSize=128;
    Early_Stop_patience=10;
    options = trainingOptions('adam',...
        'MaxEpochs',2000, ...
        'Verbose',false, ...,
        'InitialLearnRate',0.001,....
        'MiniBatchSize',miniBatchSize, ...
        'OutputFcn',@(info) EarlyStopingFun(info,Early_Stop_patience));
    %'Plots','training-progress'

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % size(p)
    % size(categorical(t))
    [net_CNN,info] = trainNetwork(p',categorical(t)',layers,options);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    SizeLayer1_CNN_Net = activations(net_CNN,p',1,'outputAs','channels');
    SizeLayer2_CNN_Net = activations(net_CNN,p',2,'outputAs','channels');
    SizeLayer3_CNN_Net = activations(net_CNN,p',3,'outputAs','channels');
    SizeLayer4_CNN_Net = activations(net_CNN,p',4,'outputAs','channels');
    SizeLayer5_CNN_Net = activations(net_CNN,p',5,'outputAs','channels');
    SizeLayer6_CNN_Net = activations(net_CNN,p',6,'outputAs','channels');
    SizeLayer7_CNN_Net = activations(net_CNN,p',7,'outputAs','channels');
    feature_CNN = activations(net_CNN,p',3,'outputAs','channels');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%% GRU Network %%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    num_features_GRU=num_feature;
    num_sampels_GRU=number_of_data;
    feature=p;
    for i=1:1:num_sampels_GRU
        feature_cell(i,:)=mat2cell(feature(i,:)',num_features_GRU,1);
    end

    inputSize = num_features_GRU;

    numHiddenUnits = 40;
    numClasses = numel(unique(t));

    layers = [ ...
        sequenceInputLayer(inputSize)
        gruLayer(numHiddenUnits,'OutputMode','last')
        dropoutLayer(0.2)
        gruLayer(numHiddenUnits,'OutputMode','last')
        dropoutLayer(0.2)
        fullyConnectedLayer(numClasses)
        selfAttentionLayer(8, 64, 'Name', 'self_attention')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    maxEpochs = 2000;
    miniBatchSize = 128;
    Early_Stop_patience=10;

    options = trainingOptions('adam', ...
        'ExecutionEnvironment','cpu', ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',0.001,....
        'Verbose',false, ...
        'OutputFcn',@(info) EarlyStopingFun(info,Early_Stop_patience));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    target_cat=categorical(t);

    net_GRU = trainNetwork(feature_cell,target_cat,layers,options);
    SizeLayer1_GRU_Net = activations(net_GRU,feature_cell,1,'outputAs','channels');
    SizeLayer2_GRU_Net = activations(net_GRU,feature_cell,2,'outputAs','rows');
    SizeLayer3_GRU_Net = activations(net_GRU,feature_cell,3,'outputAs','rows');
    SizeLayer4_GRU_Net = activations(net_GRU,feature_cell,4,'outputAs','rows');
    SizeLayer5_GRU_Net = activations(net_GRU,feature_cell,5,'outputAs','rows');
    SizeLayer6_GRU_Net = activations(net_GRU,feature_cell,6,'outputAs','rows');
    SizeLayer7_GRU_Net = activations(net_GRU,feature_cell,7,'outputAs','rows');
    SizeLayer8_GRU_Net = activations(net_GRU,feature_cell,8,'outputAs','rows');
    SizeLayer9_GRU_Net = activations(net_GRU,feature_cell,9,'outputAs','rows');
    feature_GRU = activations(net_GRU,feature_cell,5,'outputAs','rows');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%% Feature Selection %%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    feature_CNN=double(cell2mat(feature_CNN))';
    feature_total=[feature_CNN feature_GRU];

    %%%%%%%%%%% Apply NCA features Rankings
    m=fscnca(repmat(feature_total,5,1),categorical(repmat(t,5,1))); %Since INCA Need more Samples to learn we increase samples by repeating
    feature_weights_NCA=m.FeatureWeights;

    %%%%%%%%%%% Rank features based on Redundancy
    [idx,feature_weights_redundancy] = fscmrmr(repmat(feature_total,5,1),categorical(repmat(t,5,1))); %Since INCAR Need more Samples to learn we increase samples by repeating

    %%%%%%%%%%% Combine NCA and Redundancy weigth Ranks
    feature_weights_total=feature_weights_NCA+feature_weights_redundancy';


    %%%%%%%%%%% Sort features and Select features by Higher Ranks
    [data_sort ind_sort]=sort(feature_weights_total,'descend');

    %%%%%%%%%%% Select features by Higher Ranks in ind_sort
    fetures_INCA=feature_total(:,ind_sort(1:30));



    figure(100)
    stem(feature_weights_total)
    xlabel('Feature Number')
    ylabel('Scores')
    title('INCA Feature Scores (Befor Sorting)')

    % figure(101)
    % bar(data_sort)
    % xlabel('Feature Number')
    % ylabel('Scores')
    % title('INCA Feature Scores (Sorted)')


    ind_sort_Categorical = categorical(ind_sort);
    ind_sort_Categorical = reordercats(ind_sort_Categorical,string(categorical(ind_sort)));

    figure(100)
    bar(ind_sort_Categorical, data_sort, 'BarWidth', 0.5)
    xlabel('Feature Number')
    ylabel('Scores')
    title('INCA Feature Scores (Sorted)')


    % Set Axis Options
    ax = gca; % Get Axis
    ax.XAxis.FontWeight = 'bold'; % Set Bold Style for Texts
    ax.XAxis.FontSize = 12;

    % Feature_Selection=[data_sort;ind_sort];
    % fetures_MRMR=feature_total(:,ind_sort(1:30));


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Classification for DOA Estimation %%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    p=fetures_INCA';
    for i=1:number_of_data
        trainD(:,:,:,i)=p(:,i);
    end
    targetD=categorical(t');

    num_feature_INCA=size(fetures_INCA,2);
    layers = [
        imageInputLayer([num_feature_INCA 1 1],'Name','input') % (num_feature)X1X1 refers to number of features per sample
        reluLayer('Name','Relu1')
        maxPooling2dLayer(1,'Name','Pooling1')
        fullyConnectedLayer(80,'Name','fc1') % number of neurons in next FC hidden layer
        fullyConnectedLayer(80,'Name','fc2') % number of neurons in next FC hidden layer
        fullyConnectedLayer(ClassNum,'Name','fc3') % number of neurons in next output layer (number of output classes)
        softmaxLayer('Name','Softmax')
        classificationLayer('Name','Classification')];
    lgraph = layerGraph(layers);
    % figure(1)
    % plot(lgraph)

    miniBatchSize =128;
    Early_Stop_patience=10;
    options = trainingOptions('adam',...  %solverName — Solver for training network'sgdm' 'rmsprop' 'adam'
        'MaxEpochs',2000, ...
        'Verbose',false, ...
        'InitialLearnRate',0.001,....
        'MiniBatchSize',miniBatchSize, ...
        'OutputFcn',@(info) EarlyStopingFun(info,Early_Stop_patience));
    %'Plots','training-progress'

    net2 = trainNetwork(trainD,targetD',layers,options);

    predictedLabels = classify(net2,trainD)';


    out2=nominal(predictedLabels);
    out_mont(q,:)=str2num(char(out2))';


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end


%%%%%%%%%%%%%%%%  F-estimate with SVR
tic
out_mont_svr=zeros(size(tt_SVR));
for j=1:mont_carlo
    F=[(real(y))';(imag(y))'];
    model1 = fitrsvm(F',tt_SVR(1,:)');
    out_svr1 = predict(model1,F');
    model2 = fitrsvm(F',tt_SVR(2,:)');
    out_svr2 = predict(model2,F');
    model3 = fitrsvm(F',tt_SVR(3,:)');
    out_svr3 = predict(model3,F');
    out_svr=[out_svr1';out_svr2';out_svr3'];
    out_mont_svr=out_svr+out_mont_svr;
end

out_mont_svr=out_mont_svr./mont_carlo;


%%%%%%%%%%% G-plot estimated Doa
figure(1)
subplot(2,2,1)
plot(tt_SVR(1,:),'r','LineWidth',2)
hold on
plot(tt_SVR(2,:),'b','LineWidth',2)
hold on
plot(tt_SVR(3,:),'g','LineWidth',2)
hold on
plot(out_mont(1,:),'og','MarkerSize',3,'LineWidth',0.8,'MarkerFaceColor','g')
hold on
plot(out_mont(2,:),'+r','MarkerSize',3.5,'LineWidth',1,'MarkerFaceColor','k')
hold on
plot(out_mont(3,:),'*b','MarkerSize',3,'LineWidth',0.8,'MarkerFaceColor','b')
lgd=legend({'$\theta$1','$\theta$2','$\theta$3','${\theta}1hat$','${\theta}$2hat','${\theta}$3hat'},'interpreter','latex','Location','southeast');
lgd.FontSize = 9;
lgd.NumColumns = 2;
lgd.Orientation = 'vertical';
lgd.ItemTokenSize = [6 6];
% lgd.FontName = 'Tahoma';
% lgd.FontWeight = 'bold';
lgd.TextColor = [0 0 0];  % Black
title({'CNN-ABGRU-INCA', '(Proposed)'},'FontSize',9)
xlim([1 45])
ylim([-30 30])
% xticks(0:5:45)
xticks(0:10:40)
yticks(-30:10:30)
% ax=gca;
% ax.FontSize=9;

%%%%%%%%%%%plot estimated Doa Errore
er=tt_SVR-out_mont;
subplot(2,2,2)
plot(er(1,:),'og','MarkerSize',3,'LineWidth',0.8,'MarkerFaceColor','g')
hold on
plot(er(2,:),'+r','MarkerSize',3.5,'LineWidth',1,'MarkerFaceColor','k')
hold on
plot(er(3,:),'*b','MarkerSize',3,'LineWidth',0.8,'MarkerFaceColor','b')
% title('Mean errors for CNN-ABGRU-INCA (Proposed)')
lgd=legend({'Error$\theta$1','Error$\theta$2','Error$\theta$3'},'interpreter','latex','Location','southeast');
lgd.FontSize = 9;
lgd.NumColumns = 1;
lgd.Orientation = 'vertical';
lgd.ItemTokenSize = [6 6];
% lgd.FontName = 'Tahoma';
% lgd.FontWeight = 'bold';
lgd.TextColor = [0 0 0];  % Black
title({'Mean errors for', 'CNN-ABGRU-INCA(Proposed)'},'FontSize',9)
ylim([-5 5])
xlim([1 45])
% xticks(0:5:45)
xticks(0:10:40)
yticks(-5:5:5)
% ax=gca;
% ax.FontSize=9;

%%%%%%%%%%%plot estimated Doa with SVR
figure(1)
subplot(2,2,3)
plot(tt_SVR(1,:),'r','LineWidth',2)
hold on
plot(tt_SVR(2,:),'b','LineWidth',2)
hold on
plot(tt_SVR(3,:),'g','LineWidth',2)
hold on
plot(out_mont_svr(1,:),'og','MarkerSize',3,'LineWidth',0.8,'MarkerFaceColor','g')
hold on
plot(out_mont_svr(2,:),'+r','MarkerSize',3.5,'LineWidth',1,'MarkerFaceColor','k')
hold on
plot(out_mont_svr(3,:),'*b','MarkerSize',3,'LineWidth',0.8,'MarkerFaceColor','b')
lgd=legend({'$\theta$1','$\theta$2','$\theta$3','${\theta}1hat$','${\theta}$2hat','${\theta}$3hat'},'interpreter','latex','Location','southeast');
lgd.FontSize = 9;
lgd.NumColumns = 2;
lgd.Orientation = 'vertical';
lgd.ItemTokenSize = [6 6];
% lgd.FontName = 'Tahoma';
% lgd.FontWeight = 'bold';
lgd.TextColor = [0 0 0];  % Black
title('SVR','FontSize',9)
xlim([1 45])
ylim([-30 30])
% xticks(0:5:45)
xticks(0:10:40)
yticks(-30:10:30)
% ax=gca;
% ax.FontSize=9;

%%%%%%%%%%%plot estimated Doa with MUSIC
est_DOAs_matlab=est_DOAs_matlab';

figure(1)
subplot(2,2,4)
plot(tt_SVR(1,:),'r','LineWidth',2)
hold on
plot(tt_SVR(2,:),'b','LineWidth',2)
hold on
plot(tt_SVR(3,:),'g','LineWidth',2)
hold on
plot(est_DOAs_matlab(1,:),'og','MarkerSize',3,'LineWidth',0.8,'MarkerFaceColor','g')
hold on
plot(est_DOAs_matlab(2,:),'+r','MarkerSize',3.5,'LineWidth',1,'MarkerFaceColor','k')
hold on
plot(est_DOAs_matlab(3,:),'*b','MarkerSize',3,'LineWidth',0.8,'MarkerFaceColor','b')
lgd=legend({'$\theta$1','$\theta$2','$\theta$3','${\theta}1hat$','${\theta}$2hat','${\theta}$3hat'},'interpreter','latex','Location','southeast');
lgd.FontSize = 9;
lgd.NumColumns = 2;
lgd.Orientation = 'vertical';
lgd.ItemTokenSize = [6 6];
% lgd.FontName = 'Tahoma';
% lgd.FontWeight = 'bold';
lgd.TextColor = [0 0 0];  % Black
title('MUSIC','FontSize',9)
xlim([1 45])
ylim([-30 30])
% xticks(0:5:45)
xticks(0:10:40)
yticks(-30:10:30)
% ax=gca;
% ax.FontSize=9;

% Set Fifure Size
set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
pos(3) = 4.5;  % Set Figure Width
pos(4) = 5.0;  % Set Figure Height
set(gcf, 'Position', pos);


set(gcf, 'PaperUnits', 'inches', 'PaperPosition', pos);
print(gcf, 'output2.png', '-dpng', '-r600');



%%%%%%%%%%% G-plot estimated Doa
figure(2)
subplot(3,1,1)
plot(tt_SVR(1,:),'r','LineWidth',2)
hold on
plot(tt_SVR(2,:),'b','LineWidth',2)
hold on
plot(tt_SVR(3,:),'g','LineWidth',2)
hold on
plot(out_mont(1,:),'og','MarkerSize',4.5,'LineWidth',0.8,'MarkerFaceColor','g')
hold on
plot(out_mont(2,:),'+r','MarkerSize',6,'LineWidth',1,'MarkerFaceColor','k')
hold on
plot(out_mont(3,:),'*b','MarkerSize',4.5,'LineWidth',0.8,'MarkerFaceColor','b')
legend('Theta 1','Theta 2','Theta 3','Theta 1 Predicted','Theta 2 Predicted','Theta 3 Predicted','Location','southeast','NumColumns',2)
xlim([1 45])
ylim([-30 30])
title('CNN-ABGRU-INCA (Proposed)')


%%%%%%%%%%%plot estimated Doa with SVR
figure(2)
subplot(3,1,2)
plot(tt_SVR(1,:),'r','LineWidth',2)
hold on
plot(tt_SVR(2,:),'b','LineWidth',2)
hold on
plot(tt_SVR(3,:),'g','LineWidth',2)
hold on
plot(out_mont_svr(1,:),'og','MarkerSize',4.5,'LineWidth',0.8,'MarkerFaceColor','g')
hold on
plot(out_mont_svr(2,:),'+r','MarkerSize',6,'LineWidth',1,'MarkerFaceColor','k')
hold on
plot(out_mont_svr(3,:),'*b','MarkerSize',4.5,'LineWidth',0.8,'MarkerFaceColor','b')
legend('Theta 1','Theta 2','Theta 3','Theta 1 Predicted','Theta 2 Predicted','Theta 3 Predicted','Location','southeast','NumColumns',2)
xlim([1 45])
ylim([-30 30])
title('SVR')


%%%%%%%%%%%plot estimated Doa with MUSIC
figure(2)
subplot(3,1,3)
plot(tt_SVR(1,:),'r','LineWidth',2)
hold on
plot(tt_SVR(2,:),'b','LineWidth',2)
hold on
plot(tt_SVR(3,:),'g','LineWidth',2)
hold on
plot(est_DOAs_matlab(1,:),'og','MarkerSize',4.5,'LineWidth',0.8,'MarkerFaceColor','g')
hold on
plot(est_DOAs_matlab(2,:),'+r','MarkerSize',6,'LineWidth',1,'MarkerFaceColor','k')
hold on
plot(est_DOAs_matlab(3,:),'*b','MarkerSize',4.5,'LineWidth',0.8,'MarkerFaceColor','b')
legend('Theta 1','Theta 2','Theta 3','Theta 1 Predicted','Theta 2 Predicted','Theta 3 Predicted','Location','southeast','NumColumns',2)
xlim([1 45])
ylim([-30 30])
title('MUSIC')


%%%%%%%%%%%plot estimated Doa Errore
figure(3)
er=tt_SVR-out_mont;
subplot(2,1,1)
plot(er(1,:),'og','MarkerSize',4.5,'LineWidth',0.8,'MarkerFaceColor','g')
hold on
plot(er(2,:),'+r','MarkerSize',6,'LineWidth',1,'MarkerFaceColor','k')
hold on
plot(er(3,:),'*b','MarkerSize',4.5,'LineWidth',0.8,'MarkerFaceColor','b')
title('Mean errors for CNN-ABGRU-INCA (Proposed)')
legend('Error Theta 1','Error Theta 2','Error Theta 3','Location','southeast','NumColumns',2)
ylim([-5 5])
xlim([1 45])

subplot(2,1,2)
stem(er(1,:),'og','MarkerSize',4.5,'LineWidth',0.8,'MarkerFaceColor','g')
hold on
stem(er(2,:),'+r','MarkerSize',6,'LineWidth',1,'MarkerFaceColor','k')
hold on
stem(er(3,:),'*b','MarkerSize',4.5,'LineWidth',0.8,'MarkerFaceColor','b')
title('Mean errors for CNN-ABGRU-INCA (Proposed)')
legend('Error Theta 1','Error Theta 2','Error Theta 3','Location','southeast','NumColumns',2)
ylim([-5 5])
xlim([1 45])





