Search.setIndex({docnames:["buildyourown","changelog","docs","docs_filters","docs_models","docs_optimizer","docs_retina","docs_streams","docs_tests","examples","filters","filters_Conv3d","index","installation","model","model_McIntosh","model_ln","model_retina","pytorch_basics","quickstart","usage"],envversion:52,filenames:["buildyourown.rst","changelog.rst","docs.rst","docs_filters.rst","docs_models.rst","docs_optimizer.rst","docs_retina.rst","docs_streams.rst","docs_tests.rst","examples.rst","filters.rst","filters_Conv3d.rst","index.rst","installation.rst","model.rst","model_McIntosh.rst","model_ln.rst","model_retina.rst","pytorch_basics.rst","quickstart.rst","usage.rst"],objects:{"":{convis:[2,0,0,"-"]},"convis.base":{Layer:[2,1,1,""],Model:[2,3,1,""],Output:[2,1,1,""],Runner:[2,1,1,""],prepare_input:[2,4,1,""],shape:[2,4,1,""]},"convis.base.Layer":{clear_state:[2,2,1,""],compute_loss:[2,2,1,""],cpu:[2,2,1,""],cuda:[2,2,1,""],get_all:[2,2,1,""],get_parameters:[2,2,1,""],get_state:[2,2,1,""],load_parameters:[2,2,1,""],m:[2,3,1,""],optimize:[2,2,1,""],p:[2,3,1,""],parse_config:[2,2,1,""],plot_impulse:[2,2,1,""],plot_impulse_space:[2,2,1,""],pop_all:[2,2,1,""],pop_optimizer:[2,2,1,""],pop_parameters:[2,2,1,""],pop_state:[2,2,1,""],push_all:[2,2,1,""],push_optimizer:[2,2,1,""],push_parameters:[2,2,1,""],push_state:[2,2,1,""],requires_grad_:[2,2,1,""],retrieve_all:[2,2,1,""],run:[2,2,1,""],s:[2,3,1,""],save_parameters:[2,2,1,""],set_parameters:[2,2,1,""],set_state:[2,2,1,""],store_all:[2,2,1,""]},"convis.base.Output":{array:[2,2,1,""],mean:[2,2,1,""],plot:[2,2,1,""]},"convis.base.Runner":{optimize:[2,2,1,""],run:[2,2,1,""],start:[2,2,1,""],stop:[2,2,1,""]},"convis.filters":{Conv1d:[3,1,1,""],Conv2d:[3,1,1,""],Conv3d:[10,1,1,""],Delay:[3,1,1,""],Diff:[3,1,1,""],NLRectify:[10,1,1,""],NLRectifyScale:[10,1,1,""],NLRectifySquare:[10,1,1,""],NLSquare:[10,1,1,""],RF:[10,1,1,""],SmoothConv:[3,1,1,""],Sum:[3,1,1,""],TimePadding:[3,1,1,""],VariableDelay:[3,1,1,""],retina:[6,0,0,"-"],spiking:[10,0,0,"-"],sum:[3,4,1,""]},"convis.filters.Conv1d":{exponential:[3,2,1,""],set_weight:[3,2,1,""]},"convis.filters.Conv2d":{set_weight:[3,2,1,""]},"convis.filters.Conv3d":{exponential:[10,2,1,""],gaussian:[10,2,1,""],highpass_exponential:[10,2,1,""],set_weight:[10,2,1,""]},"convis.filters.retina":{Bipolar:[6,1,1,""],FullConvolutionOPLFilter:[6,1,1,""],GanglionInput:[6,1,1,""],GanglionSpiking:[6,1,1,""],HalfRecursiveOPLFilter:[6,1,1,""],OPL:[6,1,1,""],RecursiveOPLFilter:[6,1,1,""],SeperatableOPLFilter:[6,1,1,""]},"convis.filters.spiking":{FitzHughNagumo:[10,1,1,""],HogkinHuxley:[10,1,1,""],IntegrativeMotionSensor:[10,1,1,""],Izhikevich:[10,1,1,""],LeakyIntegrateAndFireNeuron:[10,1,1,""],Poisson:[10,1,1,""],RefractoryLeakyIntegrateAndFireNeuron:[10,1,1,""]},"convis.filters.spiking.Izhikevich":{load_parameters_by_name:[10,2,1,""]},"convis.kernels":{ExponentialKernel:[3,1,1,""]},"convis.models":{Dict:[4,1,1,""],L:[4,1,1,""],LN:[4,1,1,""],LNCascade:[4,1,1,""],LNFDLNF:[4,1,1,""],LNFDSNF:[4,3,1,""],LNLN:[4,1,1,""],List:[4,1,1,""],McIntosh:[4,1,1,""],Parallel:[4,1,1,""],RF:[4,1,1,""],Retina:[4,1,1,""],Sequential:[4,1,1,""],make_parallel_sequential_model:[4,4,1,""],make_sequential_parallel_model:[4,4,1,""]},"convis.models.List":{append:[4,2,1,""],extend:[4,2,1,""]},"convis.optimizer":{CautiousLBFGS:[5,1,1,""],FiniteDifferenceGradientOptimizer:[5,1,1,""]},"convis.optimizer.CautiousLBFGS":{step:[5,2,1,""]},"convis.optimizer.FiniteDifferenceGradientOptimizer":{step:[5,2,1,""]},"convis.retina":{Retina:[6,1,1,""]},"convis.retina_virtualretina":{RetinaConfiguration:[6,1,1,""],deriche_filter_density_map:[6,4,1,""]},"convis.retina_virtualretina.RetinaConfiguration":{get:[6,2,1,""],read_json:[6,2,1,""],set:[6,2,1,""],write_json:[6,2,1,""],write_xml:[6,2,1,""]},"convis.samples":{SampleGenerator:[2,1,1,""],StimulusSize:[2,1,1,""],generate_sample_data:[2,4,1,""],moving_bar:[2,4,1,""],moving_grating:[2,4,1,""],random_checker_stimulus:[2,4,1,""]},"convis.samples.SampleGenerator":{generate:[2,2,1,""]},"convis.streams":{ImageSequence:[7,1,1,""],InrImageFileStreamer:[7,1,1,""],InrImageStreamWriter:[7,1,1,""],InrImageStreamer:[7,1,1,""],MNISTStream:[7,1,1,""],NumpyReader:[7,1,1,""],PoissonMNISTStream:[7,1,1,""],ProcessingStream:[7,1,1,""],PseudoNMNIST:[7,1,1,""],RandomStream:[7,1,1,""],RepeatingStream:[7,1,1,""],SequenceStream:[7,1,1,""],Stream:[7,1,1,""],TimedResampleStream:[7,1,1,""],TimedSequenceStream:[7,1,1,""]},"convis.streams.ImageSequence":{get:[7,2,1,""]},"convis.streams.InrImageStreamer":{get:[7,2,1,""]},"convis.streams.MNISTStream":{get:[7,2,1,""],reset:[7,2,1,""]},"convis.streams.NumpyReader":{get:[7,2,1,""]},"convis.streams.PoissonMNISTStream":{get:[7,2,1,""],reset:[7,2,1,""]},"convis.streams.RandomStream":{get:[7,2,1,""]},"convis.streams.RepeatingStream":{get:[7,2,1,""]},"convis.streams.SequenceStream":{get:[7,2,1,""]},"convis.streams.Stream":{get:[7,2,1,""]},"convis.utils":{extend_to_match:[2,4,1,""],make_tensor_5d:[2,4,1,""],plot:[2,4,1,""],plot_5d_matshow:[2,4,1,""],plot_5d_time:[2,4,1,""],plot_tensor:[2,4,1,""],plot_tensor_with_channels:[2,4,1,""],subtract_tensors:[2,4,1,""]},"convis.variable_describe":{OrthographicWrapper:[2,1,1,""],animate:[2,4,1,""],animate_double_plot:[2,4,1,""],animate_to_html:[2,4,1,""],animate_to_video:[2,4,1,""],describe:[2,4,1,""],plot_3d_tensor_as_3d_plot:[2,4,1,""]},"convis.variables":{Parameter:[2,1,1,""],Variable:[2,3,1,""],VirtualParameter:[2,1,1,""],create_Ox_from_torch_iterator_dicts:[2,4,1,""],create_context_O:[2,4,1,""],create_hierarchical_dict:[2,4,1,""],create_hierarchical_dict_with_nodes:[2,4,1,""]},"convis.variables.Parameter":{requires_grad_:[2,2,1,""]},convis:{base:[2,0,0,"-"],filters:[3,0,0,"-"],kernels:[3,0,0,"-"],layers:[4,0,0,"-"],models:[4,0,0,"-"],optimizer:[5,0,0,"-"],retina:[6,0,0,"-"],retina_virtualretina:[6,0,0,"-"],samples:[2,0,0,"-"],set_pixel_per_degree:[2,4,1,""],set_steps_per_second:[2,4,1,""],streams:[7,0,0,"-"],tests:[8,0,0,"-"],utils:[2,0,0,"-"],variable_describe:[2,0,0,"-"],variables:[2,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function"},terms:{"0x7f99d1d88750":2,"10x10":3,"1px":9,"1x1x1000x1x1":20,"2nd":5,"4th":15,"5ms":20,"9x9":3,"case":[2,4,6,9,11,17,18,20],"class":[1,3,4,5,6,7,9,10,11,12,15,17,18],"default":[1,2,3,4,6,7,9,10,16,17,20],"final":[6,9,11,15,18],"float":[2,7,10,11,18,20],"function":[1,2,3,4,6,7,9,10,11,12,15,18,20],"import":[0,2,3,4,5,6,7,9,10,15,19],"int":[2,3,4,7,11,18],"long":[1,2,3,7,9,20],"new":[1,2,3,5,6,7,10,17,20],"public":[10,12],"return":[1,2,3,4,5,6,7,9,10,11,15,18,20],"short":[0,2],"static":[4,6,17],"super":[2,11,15,18],"transient":6,"true":[1,2,3,4,5,6,7,9,10,11,15,18,20],"try":9,"var":2,"while":[6,10,18],And:[6,15],But:[5,6,9,20],DVS:1,For:[3,4,9,10,15,20],One:20,The:[0,1,3,4,5,6,7,9,10,11,12,15,16,17,18,20],Then:[0,9,19],There:[9,10,16,20],These:[2,4],Use:[3,10],Using:[2,4],With:[2,5],__call__:7,__dict__:2,__init__:[2,11,15,18],__iteritem:20,_all:[2,20],_debug:2,_get_default_resolut:1,_izhikevich_paramet:10,_newvari:2,_optim:5,_search:[2,20],_self:2,_time:[4,6],_use_cuda:[2,11,15,18],a_0:20,a_1:20,a_optim:5,a_var:2,a_video:9,abl:10,about:[1,2,7,12,15],abov:[2,6,20],abs:[10,20],absolut:7,acceler:20,accept:[2,3,6,10,15,18,20],access:9,accord:2,accordingli:[9,18],account:[3,10,12],accur:20,achiev:[3,10],act:10,activ:[4,6,7,9,12,15,16],actual:[2,15,17],adadelta:[2,20],adagrad:[2,20],adam:[2,9,15,20],adapt:[2,6,20],add:[2,9,10,16,18],add_lay:[3,4,10,16],add_modul:[15,20],added:[1,2,3,4,6,10,12],adding:[2,3,10,20],addit:[2,3,5,7,10,15,18,20],addition:3,adjust_pad:[3,10],advanc:[4,7,15,20],advantag:10,after:[2,3,7,9,10,20],again:[2,5,6,9],algorithm:[5,9],alia:[2,4],align:[1,2,3],all:[1,2,3,5,6,7,9,10,12,15,17,18,19],allow:[2,10],almost:[2,20],along:[2,7],alpha:[0,4,6],alreadi:[1,4,7,11,12,18],also:[0,2,3,4,6,9,10,14,16,17,20],alter:6,altern:[4,5,7,20],alwai:[1,2,4,12,15,20],amount:[1,6,9,10],ampfeedback:6,amphas:0,ampinputcurr:6,amplif:[3,20],amplification__hz:6,amplitud:10,ampopl:6,analysi:20,angl:2,ani:[3,6,7,9,10,15,20],aniamt:2,anihil:0,anim:2,animate_double_plot:2,animate_to_html:2,animate_to_video:2,annot:2,announc:12,anoth:[5,6,10,20],answer:12,anti:0,anyth:2,api:12,appar:[11,18],append:[3,4],appli:[3,4,6,10,15,20],appreci:12,approach:15,appropri:20,approx:2,approxim:[2,3,5,10],apt:12,arbitrari:7,arbitrarili:1,area:2,arg:[2,3,4,10],argument:[1,2,3,4,5,7,9,10,20],arithm:10,arleo:12,around:[6,9,10],arrai:[1,2,3,6,7,9,10,11,17,18,19,20],arxiv:4,as_paramet:2,asgd:2,ask:14,assign:[2,17,20],assum:[0,5,15],attribut:[2,3,4,6,7,9,10,20],autograd:[2,11,12,20],autom:[7,12],automat:[2,3,4,6,10,12,18,20],autopad:[1,3,4,10,11,18],autopad_mod:3,avail:[1,2,4,10,14,15,16,20],averag:2,avoid:[1,3],axi:[0,2,4,6],axon:10,azimuth:2,b_0:20,baccu:[4,15],back:2,backend:2,backpropag:[18,20],backup:5,backward:[18,20],bad:9,bar:2,bar_direct:2,bar_po:2,bar_v:2,bar_width:2,base:[1,3,4,6,7,10,12,15,17,20],basic:[3,6,10,12],batch:[2,3,4,9,15,20],befor:[0,2,5,7,15,17],begin:2,behav:[2,18],behavior:[2,10],behaviour:[2,3,6,20],being:[2,6],below:10,best:[9,12,17,20],better:4,between:[2,3,5,6,7,10,11,15,18],bia:[3,4,10,11,15,18],bichler:7,big:2,bin:[2,9,19,20],binari:[2,7,10],bio:4,biolog:[4,6],bip:[6,17],bipolar:[2,4,6,17,20],bipolar_conv2d_weight:20,bipolar_g_leak:[2,20],bipolar_input_amp:20,bipolar_lambda_amp:20,bipolar_tau:20,block:0,blur:6,bool:[2,3,4,6,7,10],border:[2,9,11,18],border_v:2,both:[1,2,4,6,7,9,10,11,17,18,20],bottom:[3,10],bound:2,broadcast:[2,18,20],broken:9,broyden:20,buffer:[2,3,7,18,20],bug:1,build:[4,12],built:[15,18],calcul:6,call:[1,2,3,18,20],callabl:[5,20],came:[4,19],can:[0,1,2,3,4,5,6,7,9,10,11,12,14,15,17,18,19,20],capabl:4,captur:9,care:6,carri:7,cascad:[3,4,6,9,10,16],cat:[4,7,11,15,18],cat_1:4,cat_2:4,cat_3:4,caus:[2,3],cautiouslbfg:5,cdot:10,ceil:[11,18],cell:[4,6,7,9,10,15,16,20],center:[1,2,3,6,7],center_:6,center_g:6,certain:[5,6,9,19],ch2:10,chang:[1,2,3,4,5,6,7,10,12,17,20],changelog:12,channel:[2,3,4,6,7,9,10,15,16,19,20],cheat:9,checker:2,checker_s:2,children:20,chirp:15,choic:2,choos:[5,7,9,19],chosen:7,chunk:[2,3,4,6,9,15,18,20],chunk_length:18,circular:6,cite:12,clamp:4,clear_stat:[2,3,20],clearli:9,close:[2,5,9],closur:[5,20],code:[0,2,3,4,6,7,9,10,15,19,20],coeffici:6,collaps:3,collect:[2,3,9,18,20],collis:2,color:[2,4,7,15,20],color_channel:7,colorbar:20,colour:20,com:[1,12,13],combin:[2,3,6,7,10,15],come:[2,9,12,20],commit:12,compar:[2,4,6,9,20],compat:6,complet:[2,4,6,9,17,18,20],complex:[4,9],complic:12,compon:3,compress:9,comput:[1,2,3,4,6,7,9,10,12,15,17],compute_loss:[2,9,20],concaten:[0,2,3,4,6],concentr:10,conda:12,condit:[11,18],conf:[2,6,17],config:[2,6],config_default:2,config_fil:9,config_kei:2,configur:[1,2,6,9],confus:5,consecut:[2,3],consid:[3,10,12,15],consist:2,consol:2,constant:[2,3,6,10,11,18],constructor:[4,20],consum:9,contain:[1,2,3,4,7,10,15,17,20],content:12,context:[2,7],contin:[11,18],continu:[1,2,4],contour:2,contour_cmap:2,contourf_cmap:2,contrast:[0,1,2,4,6,17],control:[3,4,6,10,17,20],conv1:[2,4],conv1d:[3,10],conv2:[2,4],conv2d:[1,3,10,20],conv3d:[1,2,3,4,6,10,12,16,20],conv:[0,2,9,10,11,15,18,20],convent:[15,20],converg:3,convers:[4,6,7,20],convert:[2,9,19,20],convi:[0,1,10,11,13,14,15,16,17,19,20],convis2:2,convlut:3,convolut:[1,2,3,6,9,11,12,14,16,17,18,20],convolv:[3,15],convolve_1:15,convolve_2:15,cook:7,coolwarm_r:2,copi:[1,2,3,6,10,20],corner:[3,7,10],correct:20,correctli:[1,3,10],correspond:[2,3,6,10,17,20],cos:0,counter:7,cours:15,cover:[3,7],covni:10,cpu:[2,3,15],creat:[0,1,2,3,4,6,7,9,10,11,15,16,17,18,20],create_context_o:2,create_hierarchical_dict:2,create_hierarchical_dict_with_nod:2,create_ox_from_torch_iterator_dict:2,creation:1,crop:7,cross:10,cuda:[2,11,13,15,18,20],current:[2,3,4,6,7,10,12],custom:[4,11,18],cut:[2,3,10],data:[3,5,7,10,11,12,15,18,20],data_fold:7,dataset:7,deal:20,decreas:[9,10],decrement:6,deep:[4,15,18],def:[2,11,15,18,20],default_:20,default_grad_en:20,default_resolut:[1,20],defin:[2,7,15],definit:7,deg:6,degre:[6,20],delai:[3,4,16],delet:2,dens:15,density__inv:6,depend:[2,3,7,9,15,20],derich:6,deriche_coeffici:6,deriche_filter_density_map:6,deriv:3,descent:[2,20],describ:[1,12],design:9,desir:[2,15],desired_outp:20,detach:[2,11,18],detect:7,deterior:9,determin:[2,7,20],determinist:7,dev:12,develop:12,deviat:[6,10],devic:[2,7],dfrac:[6,10],dicitonari:6,dict:[1,2,4],dict_with_dots_to_hierarchical_dict:2,dictionari:[2,4,6,10,17,20],did:9,didn:9,diehl:7,diff:[1,3],differ:[1,2,3,5,6,7,9,11,18,20],differenti:[6,12],digit:7,dilat:[3,10],dim:[2,3,11,15,18,20],dimens:[2,3,4,7,10,11,15,18,20],dimension:[0,9,20],dimensionless:[6,10],dir:4,direct:[2,9],directli:[1,2,5,6,10,13,18,20],disabl:[1,2],disc:2,discard:[0,15],discuss:12,disk:2,displai:[2,7],distinct:4,distribut:[2,6,10,16],doc:[1,2,17,20],docstr:12,document:[1,2],doe:[2,3,4,5,6,9,10,11,15,18,20],dog:7,doi:[4,6,12],doing:[2,9],dollfu:7,don:[1,9],done:[7,20],dot:[2,6,10],doubl:20,doubletensor:18,download:[2,7],draw:2,drawn:[6,10],driven:7,dtype:[2,7,17,20],due:[1,5],dump_patch:20,dure:[2,3],dv_:[6,17],dv_n:[6,10],dynam:10,each:[2,3,4,6,7,9,10,15,17,20],earli:12,easi:[2,7,18],easier:[1,4,16,20],edg:[3,10],effect:[3,10],effici:[3,6],effort:4,either:[2,4,6,7,9,11,12,18,20],element:[2,6],els:[2,11,15,18],embed:2,empti:[3,6,10],enabl:[2,3,6,10],encod:7,encourag:12,end:[2,4,7],end_valu:2,enhanc:[7,12],enough:3,entir:20,entri:[2,4,6],enumer:4,epfl:10,equal:[2,6,10],equat:6,equival:2,error:[9,20],especi:20,essenti:4,estim:5,eta_v:[6,10],etc:[2,4,20],eugen:10,eval:20,evalu:[2,9,20],even:[2,3,20],event:7,everi:20,everyth:[2,9],evolut:10,exactli:[2,3,10,15,20],examin:9,exampl:[0,1,2,3,4,6,7,10,12,20],except:3,execut:[0,2,4,5,6,7,9,20],exemplari:2,exist:7,exp:4,expand:3,expect:2,experiment:7,experimentalist:[9,12,19],explor:[2,17,20],exponenti:[3,6,10],exponential_filter_1d:10,exponential_filter_5d:2,exponentialkernel:3,express:7,extend:[2,3,4,10,12],extend_to_match:2,extens:12,extent:[3,10,16],extract:7,factor:[6,7],fall:[2,10],fals:[1,2,3,4,6,7,9,10,11,15,18,20],far:9,fast:20,faster:[10,20],fatherretina:6,featur:[3,12,18,20],fed:2,feed:[1,9],feedback:[4,6,16],feedback_length:4,feel:4,few:[2,9,10,18,20],fft:10,field:[0,2,3,4,6,12,16,20],figsiz:15,figur:[2,4,6,9,10,15,19,20],file:[2,6,7,9],file_ref:7,filenam:[2,6,7],filetyp:2,fill:[2,3,20],filter:[1,2,4,7,9,11,12,15,16,18,19,20],filter_2_s:[4,15],filter_length:[11,18],filter_s:[4,15],find:[2,7,9,12,14],finit:[5,9],finitedifferencegradientoptim:5,fire:[6,7,9,10],first:[1,2,3,4,5,6,9,10,11,15,17,18,19,20],first_fram:3,fit:[2,3,5,10,12,15,16],fittabl:3,fitted_model:9,fitzhugh:10,fitzhughnagumo:10,five:20,fix:[1,3,7,12,20],flag:[1,12,20],flat:2,flatten:[10,20],fletcher:20,flicker:2,flip:[1,3,10],float32:[2,17,20],float64:7,floor:[11,18],fninf:12,focu:18,folder:9,follow:[6,7,10,12,15,20],fork:12,format:[2,9,19],formula:[6,17],forward:[2,4,11,15,18,20],found:[1,2,4,7,9,15,20],four:10,fovea:6,fovea__inv:6,frac:[6,10,17],frame:[2,3,7,9,10,11,18,20],frame_:7,frames_per_second:2,freq:10,frequenc:[9,10,19],frequent:6,from:[1,2,3,4,5,6,7,9,10,12,13,15,17,18,19,20],from_numpi:18,front:[3,12],frontier:7,full:[2,3,6,11,18,20],full_copi:3,fullconvolutionoplfilt:[1,6,17],fullli:6,func:2,funcanim:2,further:[2,3,10],futur:3,fuzzy:2,g_l:[6,9,10,19],g_leak:[2,20],gabor:0,gabor_kernel:0,gain:[4,6,10,17,20],gamrat:7,gang:[4,6,10,17],gang_0_input:[2,4,6,17,20],gang_0_input_f_transi:20,gang_0_input_i_0:20,gang_0_input_lambda_g:20,gang_0_input_sigma_surround:20,gang_0_input_sign:20,gang_0_input_spatial_pooling_weight:20,gang_0_input_transient_relative_weight_cent:20,gang_0_input_transient_tau_cent:20,gang_0_input_transient_weight:20,gang_0_input_v_0:20,gang_0_spik:[2,4,6,17,20],gang_0_spikes_g_l:20,gang_0_spikes_noise_sigma:20,gang_0_spikes_refr_mu:20,gang_0_spikes_refr_sigma:20,gang_0_spikes_tau:20,gang_1_input:[2,4,6,17,20],gang_1_input_f_transi:20,gang_1_input_i_0:20,gang_1_input_lambda_g:20,gang_1_input_sigma_surround:20,gang_1_input_sign:20,gang_1_input_spatial_pooling_bia:20,gang_1_input_transient_relative_weight_cent:20,gang_1_input_transient_tau_cent:20,gang_1_input_transient_weight:20,gang_1_input_v_0:20,gang_1_spik:[2,4,6,17,20],gang_1_spikes_g_l:20,gang_1_spikes_noise_sigma:20,gang_1_spikes_refr_mu:20,gang_1_spikes_refr_sigma:20,gang_1_spikes_tau:20,ganglion:[4,6,9,10,15],ganglion_spikes_off:9,ganglion_spikes_on:9,ganglioninput:[1,4,6,10,17],ganglionspik:[1,4,6,10,17],ganguli:[4,15],gauss_filter_2d:10,gaussian:[1,2,3,6,10],gca:10,gcf:7,gener:[0,2,7,12,20],generaliz:4,generate_sample_data:[2,9],get:[1,2,3,4,6,7,9,10,12,13,15,16,17,19,20],get_al:[2,5,20],get_all_compon:3,get_config:2,get_paramet:[2,17,20],get_stat:[2,20],giant:10,git:[1,12,13],github:[4,7,12,13],give:[2,3,4,7,9,16,17,20],given:3,gleak:6,global:[1,2],glue:6,goal:[2,20],goal_a:5,goal_b:5,goldfarb:20,googlegroup:12,gpu:[2,15],grad:18,gradient:[2,3,5,9,18,20],grai:[2,15],graph:[0,1,2,18],grate:[2,3,10],grayscal:7,greater:2,grid:2,ground:9,group:[3,10],guid:12,had:1,half:[0,20],halfrecursiveoplfilt:[6,17],hand:[2,6,7],handl:20,happen:20,hard:[9,11,18],has:[0,1,2,3,4,6,7,9,10,15,18,20],have:[1,2,3,5,7,9,10,11,12,14,15,18,19,20],heat:6,height:[2,3,10],help:2,helper:20,helperfunct:6,here:[0,4,5,6,7,9],hierarch:[2,18],hierarchi:[2,20],high:[6,7,10,18],higher:2,highest:7,highpass:[3,10],highpass_exponenti:[3,10],hire:[2,3,7,9,10,20],hist:20,histori:3,hodgkin:10,hogkinhuxlei:10,hold:[2,3,17],hopefulli:9,how:[2,4,6,7,9,11,15,18,20],howev:[2,3,9,10,15],htm:10,html:[2,10,20],http:[1,4,6,10,12,13,20],human:6,huth:12,huxlei:10,i_0:[6,20],idea:7,ident:[3,4,10],ieee:7,ignor:[2,7,20],imag:[2,3,4,7,9,10,15,16],imagesequ:[1,7,9],immedi:15,immun:7,implement:[1,3,4,6,10,11,15,16,17,18],imposs:2,improv:9,impuls:[1,2,3,9,10],imshow:[10,20],in_channel:[3,4,10,11,18,20],includ:[0,2,4,7,12,16,20],include_label:7,increas:[3,10],independ:[9,20],index:[2,7,9,12,19],individu:[3,4,16],inert:6,infinit:[1,9,20],influenc:10,info:7,inform:[2,4,5,6,9,15],inherit:[4,15],init:[2,5,6,7],init_i:7,init_st:20,initi:[1,2,4,5,7,11,18],inlin:[0,1,15],inp:[2,3,4,6,7,9,10,15,19,20],input:[0,1,2,3,4,5,6,7,9,10,11,15,16,18],input_:9,input_a:5,input_amp:20,input_b:5,input_luminosity_rang:6,input_st:[11,18],input_stream:[7,9],inputnernst_inhibit:20,inr:[7,9],inrimagefilestream:7,inrimagestream:7,inrimagestreamwrit:[7,9],insert:[4,7],instabl:[5,9],instal:1,instanc:5,instanci:9,instanti:9,instead:[2,6,7,16,17,18,20],instruct:12,integr:[4,6,10],integrativemotionsensor:[1,7,10],intend:5,interact:20,interest:20,interfer:2,intern:[1,2,7],interpret:[2,7],interv:2,invers:10,ion:10,ipython:9,is_color:7,iscalar:2,issu:[1,4,7,12],item:4,iter:[2,4,7,20],its:[0,1,2,4,9,10,11,15,18,20],itself:[2,4,12,20],izhikevich:10,jahuth:[1,12,13],javascript:2,journal:[4,6],jpeg:7,jpg:[7,9],json:[6,17],jump:12,jupyt:[2,9,12,20],just:[2,3,9,10,15],k_goal:20,keep:[1,2,3,10,15,20],keep_timing_info:[4,6],kei:[2,6],kernel:[2,4,6,9,10],kernel_dim:[4,9,11,18],kernel_pad:[11,18],kernel_s:[2,3,4,10,11,18],keyword:[2,3,10,20],kind:7,know:[4,6,15,18],kornprobst:[4,6],kwarg:[2,3,4,5,6,10],kwargs_depend:2,l_goal:20,label:[2,7,10,12],lam:20,lambda:[2,4,16,20],lambda_:[6,17],lambda_amp:[17,20],lambda_g:6,lane:15,larg:[2,7],larger:[2,3,6,10],last:[2,3,4,6,9,11,18,20],later:[2,20],latest:13,layer1:[2,4,15],layer1_channel:4,layer2:[4,15],layer2_channel:4,layer:[1,2,3,6,7,9,11,12,15,16,17,20],layer_2:4,layer_:4,layer_filt:6,lbfg:[5,9,15,20],lead:[1,2,9],leak:[1,2,6,10],leak__hz:6,leaki:6,leaks__hz:6,leakyintegrateandfireneuron:[1,9,10,19],learn:[2,4,7,15,18,20],least:2,leav:5,left:[3,7,10],legend:[10,15],len:[4,11,18],length:[2,3,4,7,9,11,18,19,20],less:[2,10,20],let:[4,9,12,15,20],level:[2,7,20],lgn:19,librari:[15,18],lif:10,like:[0,1,2,3,7,9,10,11,12,18,19],limit:[2,7],line:[2,4,6,9,19,20],linear:[0,2,6,10,12,14,15,17,20],linear_ramp:2,linear_readout:15,link:12,linspac:[0,10,19],list:[1,2,4,6,7,9,10,12,14,17],list_of_lay:4,littl:9,live:1,lncascad:[3,4,10,14,16],lnfdlnf:4,lnfdsnf:[4,14,16],lnln:[4,5,14,16],lnsn:16,load:[2,6,7,9,10,17],load_paramet:[2,20],load_parameters_by_nam:10,load_state_dict:20,load_xml:17,local:2,locat:[7,15],log:[6,10],logspac:10,longer:[1,17,20],look:[2,9,10,11,12,14,18,19],lose:[11,18],loss:[2,5,9,20],loss_fn:[2,20],lot:[3,6,10,20],low:[3,10],lower:9,lowpass:[3,10],lsty:2,lumin:[4,6,9],luminos:6,machin:[9,15],made:1,magic:2,magnocellular:6,maheswaranathan:[4,15],mail:12,main:[9,12],maintain:7,make:[2,3,4,6,9,10,11,16,18,20],make_input:20,make_parallel_sequential_model:4,make_sequential_parallel_model:4,make_tensor_5d:2,manag:[2,5],mani:[2,3,4,20],manipul:0,manual:[3,7,10],map:[2,4,6],masqueli:12,master:20,match:[3,7,9,10,11,15,16,18],math:[11,18],matplotlib:[0,2,3,4,6,7,10,15,19,20],matric:9,matshow:[2,19,20],max:[2,4],max_fram:[7,20],max_lin:2,max_t:[1,2,7],max_valu:7,maxim:4,maximum:2,mayor:18,mcintosh:[4,14,15],mean:[0,2,3,4,6,7,9,10,15,18,20],mean__sec:6,meaning:[3,10,16],meaningful:2,measur:20,mechan:[4,6],membran:[6,10],memori:[2,9,10,15,20],memoryconv:15,memrist:7,merg:1,meshgrid:19,method:[3,5,7,9,10,12,15,17,18,20],microsaccad:6,might:[2,3,5,7,10,16,20],millisecond:2,min:[2,4],minim:2,mirror:[3,11,18],mismatch:2,miss:[2,3,12,17,20],mnist:7,mnist_data:7,mnist_data_fold:7,mniststream:[1,7],mode:[2,3,4,9,10,19],model:[1,2,3,5,6,7,10,12,19],modifi:16,modul:[0,1,3,4,6,7,12,14,15,17,18,20],module1:2,moment:3,momentum:20,more:[2,3,4,5,7,9,10,15,17,18,20],most:[1,4,6,9,11,12,15,18,20],mostli:5,motion:7,move:[2,6,10,15,20],movement:7,moving_bar:[0,2,9],moving_gr:[2,3,4,6,9,10,15],mp4:9,much:9,multipl:[1,2,3,6,7,9,10,16,20],multipli:10,must:4,mymcintoshmodel:15,mymemoryconv:[11,18],n_center:6,n_exampl:2,nagumo:10,name:[1,2,4,5,6,9,10,20],name_sanit:2,named_children:20,named_modul:20,named_paramet:[2,20],nanodevic:7,nanotechnolog:7,natur:[4,15],navig:20,nayebi:[4,15],ndarrai:[2,7],necessari:[15,20],necessarili:9,need:[4,9,11,12,15,18,19,20],neg:[3,10],network:[7,15],neural:[4,7,9,10,15],neuroinform:12,neuromorph:[1,7],neuron:10,neuronaldynam:10,neurosci:[4,6,7],never:3,newton:[5,9,20],next:[2,3,4,7,9,10],nicer:2,nip:[4,15],nlrectifi:[3,10,16],nlrectifyscal:[3,10],nlrectifysquar:[3,4,10],nlsquar:[3,10],node:2,nois:[6,10],noise_sigma:[6,10],non:[0,2,3,4,6,15,17],none:[0,2,3,4,5,6,7,10,11,18,20],nonlinear:[2,3,6,12,14],normal:[1,2,3,7,9,10,11,15,18,20],notat:6,note:[0,3],notebook:[2,3,9,12,20],noth:2,notic:7,now:[1,4,9,11,15,18,20],npz:[2,9],num_level:2,number:[2,3,4,6,7,15,17,20],numer:[3,6,9],numerical_filt:[2,10],numpi:[0,2,3,4,6,7,9,10,15,18,19,20],numpyread:7,o1_f:10,o2_f:10,o_init:[4,6],object:[1,6,9,12],observ:2,odd:1,off:[3,6,9,10,20],offer:[4,6,10],offset:[2,7],often:[7,9],old:[2,17],older:1,olp:[6,17],omit:[2,6],onc:[2,5,7,9,20],one:[2,3,5,6,7,10,11,12,15,17,18,20],ones:[3,5,9,11,15,18,20],onli:[1,2,3,4,5,6,9,10,17,19,20],onlin:10,onto:[2,5],onward:2,open:[7,12],opencv:12,oper:[2,6,7,10,15,20],opl:[1,2,4,6,17,20],opl_filt:6,opl_opl_filter_relative_weight:20,opl_opl_filter_surround_e_tau:[17,20],opt2:20,opt:15,optim:[2,3,4,6,9,10,12,15,18],optimiz:20,option:[1,2,3,5,6,7,10,20],order:[2,4,5],ordereddict:[2,4],org:[4,6,10,12,13,20],origin:[2,6],orth:2,orthograph:2,orthographicwrapp:2,other:[0,1,2,3,4,5,7,10,12,15],otherwis:[2,3,6,9,10],our:9,ourselv:9,out1:20,out:[2,3,4,9,10,14,19,20],out_channel:[3,4,10,11,15,18],outer:[4,6],outermost:3,outlin:15,outp:[2,20],output:[0,1,2,3,4,6,7,9,10,11,12,15,16],output_only_spik:10,output_s:7,output_stream:9,outsid:6,over:[2,3,4,6,7,9,10,16,20],overview:[2,20],overwrit:2,own:[9,12,18],packag:[6,13,18,20],pad:[1,2,3,4,7,10,11,18],page:12,pair:[2,9],paper:15,parallel:[1,4],param:5,paramet:[1,2,3,4,5,6,7,9,10,15,17,18],parameter1:2,parrot:7,pars:6,parse_config:[2,9,17,20],part:[2,3,10],parvo:6,parvocellular:6,pass:[2,3,4,6,7,10,11,18],patch:0,path:[2,7,9],pattern:[3,10],pdf:[2,3,7,9,10,20],peopl:7,per:[3,5,6,9],perfectli:3,perform:[2,3,4,5,6,7,10,17,18,20],period:[1,3,10],persp_transform:2,phase:[0,10],phi:0,pip:[1,12,13],pixel:[2,3,4,6,7,9,10,20],pixel_per_degre:[2,7,20],pixel_to_degre:2,place:[2,3,10,12,20],placement:[3,10],plan:[7,20],plane:2,plastic:7,pleas:[7,12],plexiform:[4,6],plot:[0,1,2,3,4,6,7,9,10,15,19,20],plot_3d_tensor_as_3d_plot:2,plot_5d_matshow:[2,9],plot_5d_tim:[2,3,4,6,9,15],plot_impuls:[1,2,20],plot_impulse_spac:[2,9,20],plot_tensor:[1,2,20],plot_tensor_with_channel:2,plt:[0,3,4,6,9,10,15,19,20],png:[2,3,7,9,10,20],point:[6,7],poisson:[1,7,10],poissonmniststream:[1,7],polar:6,pool:[6,17],pool__deg:6,pop:2,pop_al:[2,5,20],pop_optim:[2,20],pop_paramet:[2,20],pop_stat:[2,3,20],pope:2,popul:[4,9,16],portion:[2,3,10],posit:[2,10],possibl:[1,2,3,4,7,10,20],post:7,potenti:[2,6,10,20],ppd:2,pre:7,preambl:0,preceding_attenuationmap:20,preceding_inhibit:20,preceding_v_bip:20,predetermin:7,prefer:12,prefix:[2,4],premad:9,prepar:2,prepare_input:[2,3,15],prepend:3,prependet:3,preprend:3,present:2,preserv:[2,3,10],preserve_channel:[3,10],press:[2,9],pretti:9,previou:[0,3,5,10,11,15,18],prime:7,print:[2,4,9,10,15,20],probabl:[6,7,9],process:[1,2,3,4,5,7,10,12,15,20],processingstream:[1,7],produc:[1,4,6,10,20],profil:10,proj3d:2,project:2,properti:[9,11,18],provid:[2,3,4,5,6,18],pseudo:[7,9,15],pseudocod:[3,10],pseudonmnist:[1,7],publish:15,pull:12,purpos:2,push:[2,5],push_al:[2,5,20],push_optim:[2,20],push_paramet:[2,20],push_stat:[2,3,20],put:[6,7,12],pylab:[0,2,4,6,7,10,15,19,20],pypi:1,pyplot:[3,10],python:[0,1,4,6,12,13,17],pytorch:[1,2,3,10,11,12,13,15,20],quasi:[5,20],queri:2,querlioz:7,question:12,quick:7,quickli:9,quickstart:[12,19],rad:2,radius__deg:6,rais:3,ram:20,ran:0,rand:[2,10,15],randn:[2,4,9,10,18,20],random:[2,4,6,7,9,10,18,20],random_checker_stimulu:2,random_init:[4,15],randomli:[2,6,7,10],randomstream:[7,9,20],rang:[2,6,9,10,20],rate:[2,6,7,9,10,20],rather:20,ratio:2,reachabl:2,read:[6,7],read_json:6,readi:4,readjust:10,readout:[4,15],real:9,reappli:1,recent:[1,12],recept:[0,3,4,6,12,16],reciev:[3,6,10,15],recipi:15,recognit:7,recommend:[4,5,7,9,10,12,13],record:20,recreat:[2,7],rectif:4,rectifi:[0,3,10,15],recurs:[2,3,6,12,17,20],recursiveoplfilt:[6,17],red:20,redund:2,reevalu:[2,5],refer:[2,3,17],refr:6,refr_mu:[6,10],refr_sigma:[6,10],refractori:[1,6,10],refractoryleakyintegrateandfireneuron:[1,10],regist:[2,4],register_backward_hook:20,register_buff:20,register_forward_hook:20,register_forward_pre_hook:20,register_paramet:20,register_st:[2,20],rel:[6,7,10],relat:20,relative_ampopl:6,relative_weight:6,releas:7,relev:[4,5,6],reli:5,reliabl:2,relu1:4,relu2:4,relu:[2,4,15],rememb:[3,11,18],remov:[3,6],renam:2,rep:7,repeat:[2,7],repeatingstream:7,repetit:7,replac:[1,2,3,6,10,17,20],replic:[3,10,11,18],report:12,repositori:12,repres:[2,7],request:[3,7,12],requir:[12,20],requires_grad:[2,18,20],requires_grad_:[2,20],reset:[2,3,7,10,15],reset_weight:[3,10],reshap:15,resiz:2,resize_small_figur:2,resolut:[1,3,10],respect:[2,3,20],respond:0,respons:[0,1,2,3,4,9,10,15],restor:[2,5],result:[1,2,3,4,9,10,15],retain_graph:[18,20],retin:[4,6,15],retina:[1,2,4,7,9,10,12,14,19,20],retina_bas:6,retina_config:6,retina_config_kei:2,retina_virtualretina:2,retinaconfigur:[2,6,17],retriev:[2,5,6,7],retrieve_al:[2,5,20],retroact:2,revers:3,rf_mode:[3,10],right:[3,9,10,12],roughli:[1,2,10],run:[0,1,2,3,4,5,6,7,10,12,15,18,19],runner:[2,9],s10827:[4,6],saccad:7,same:[1,2,3,4,5,6,7,9,10,11,18,19,20],sampl:[0,3,4,6,7,9,10,12,15,19],samplegener:2,save:[2,5,6,7,9,17,20],save_nam:2,save_paramet:[2,9,20],scalabl:[3,10],scalar:[4,6],scale:[2,3,6,7,10,15,20],scale_ar:2,scene:[4,15],scheme:6,scientif:12,scroll:2,scrolling_plot:2,search:[2,12,20],second:[2,4,5,6,7,10,15,20],secondari:7,seconds_to_step:2,secret:2,section:[1,2],see:[1,2,3,7,9,10,12,20],seed:2,seen:20,select:[2,20],self:[2,3,7,11,15,18],sens:[3,4,9,10],sensit:7,separ:[1,2,4,6,9,17,20],seperatableoplfilt:[6,17],sequenc:[2,4,7,9,18,19,20],sequencestream:[7,20],sequenti:[1,4],seri:[4,9,10,16,18],session:2,set:[2,3,4,5,6,7,9,10,17,20],set_al:[5,20],set_callback_argu:2,set_config:2,set_optim:[2,5,9,15,20],set_paramet:[2,17,20],set_pixel_per_degre:2,set_stat:[2,20],set_steps_per_second:2,set_weight:[0,1,3,4,10,11,15,18,20],set_xscal:10,sgd:[2,20],shanno:20,shape:[0,1,2,4,7,9,11,18,20],share:[2,7],share_memori:20,sharp:2,shortcut:6,should:[2,3,5,6,7,9,10,15,16,20],show:[0,2,3,7,9,19],shp:[2,6],side:[2,11,18],sig:[3,10],sigma0:6,sigma:6,sigma__deg:6,sigma_cent:6,sigma_surround:6,sigmasurround:6,sign:6,signal:[3,6,10,20],signal_f:10,signifi:20,silent:2,silver:6,similar:[1,2,3,4,6,9,11,18],similarli:18,simpl:[2,3,4,10,16],simplest:20,simpli:17,simplifi:10,simualtor:17,simul:[2,4,12],simultan:1,sin:[0,10,19],sinc:[7,9,15,20],singl:[2,3,4,5,6,9,10,15,16,17,20],size:[1,2,3,4,6,7,9,10,11,15,18,20],size__deg:6,skip:2,sleep:20,slice:[3,7,11,18],slice_at:7,slight:15,slow:10,slower:6,small:[1,2,4,9],smaller:[2,3,7,9,10,20],smooth:[2,3],smoothconv:[3,4],softwar:6,solut:9,some:[0,1,2,3,4,5,7,9,15,17,19,20],some_fil:17,some_inp:20,some_input:[3,4,9,10,19,20],some_model:2,some_nam:5,some_output:[10,15],some_short_input:20,someth:[1,4,11,18],sort:[2,4],sourc:[2,3,4,5,6,7,9,10,19,20],space:[2,3,4,10,15,20],span:9,spars:2,spatial:[2,3,4,6,10,16,17,20],spatial_filt:3,spatial_pool:6,spatio:[3,10,17],special:[2,20],specif:[2,3,4,6,9,10],specifi:[2,7,9],speed:[2,10],spike:[1,4,6,7,12,17],spiking_mod:10,spk:[9,19],split:[4,9,15],sps:2,squar:[0,2,3,6,9,10,20],squid:10,stabl:[1,7],stack:[2,5],stage:[4,9,15,17],stand:2,standard:[6,10],start:[2,4,5,7,9,12,15,20],state:[2,3,5,7,10,11,15,18],state_dict:[2,20],stateless:15,stdev__sec:6,step:[1,3,5,9,20],steps_per_second:20,still:[1,4,17],stimuli:[2,7,9,19],stimulu:[0,2,3,10],stimuluss:2,sting:2,stochast:20,stop:[2,20],store:[2,3,4,5,6],store_al:[2,5,20],str:[2,3,7,10],stream:[1,2,9,12,20],stride:[3,4,10],string:[2,3,4],strong:0,strongli:[0,6],structur:[2,17,20],sub:6,subclass:2,sublay:2,submit:[4,7,12],submodul:[1,2,10,12,15,18,20],submodule_submodule_variablenam:2,subplot:2,subscript:6,substr:20,subtract:[2,6],subtract_tensor:2,subunit:[15,16],sudo:12,suffici:20,suitabl:2,sum:[3,4,9,10,11,18,20],sum_0:4,suppli:[2,3,7,9,20],support:2,sure:[2,9],surfac:2,surpass:10,surround:6,surround_:6,surround_g:6,symmetr:9,system:[4,15,20],t_skip:2,t_zero:7,tab:[2,4,9,14,17,20],take:[1,2,3,7,9,10,11,15,18,20],taken:2,target:15,task:9,tau:[1,2,3,10,20],tau__sec:6,tau_cent:6,tau_surround:6,tausurround:6,tempor:[2,3,4,6,12,17,20],temporal_smooth:2,temporalhighpassfilterrecurs:10,temporallowpassfilterrecurs:10,tensor:[2,3,6,7,10,11,12,15,17,20],term:20,test:[2,12],text:2,than:[2,3,5,6,9,10,15,18,20],the_input:[0,2,3,9,10,15],theano:[2,17],thei:[1,2,3,4,5,6,7,9,15,18,19,20],them:[2,4,6,7,9,11,12,17,18,19],therefor:20,thi:[0,1,2,3,4,5,6,7,9,10,11,12,15,17,18,20],thing:4,thought:6,thread:[2,9,20],three:[2,5,7,10,18],threshold:[6,7,10],threshold__hz:6,through:[1,2,6,7,20],time:[1,2,3,4,5,6,7,9,10,11,15,16,18,19,20],time_dimens:[7,11,18],time_pad:[1,3,4,10],timedresamplestream:7,timedsequencestream:7,timepad:[3,4],timepoint:20,timeseri:2,timestep:[2,3,9,10],timing_info:[4,6],titl:[9,10],tlim:2,togeth:15,told:3,too:[1,2,9,18,20],took:[4,6],toolbox:12,top:[2,3,7,10],torch:[2,3,4,5,7,10,11,12,15,20],torchvis:7,total:4,touch:7,track:[2,15],tracker:[7,12],trail:20,train:[7,9,19,20],transact:7,transfer:10,transform:[3,6,7,10],tree:6,tri:2,trigger:6,truth:9,tupl:[2,4,6,7],ture:9,turn:[6,20],two:[0,1,2,3,4,5,7,9,10,12,15,16,20],type:[9,11,18,20],typic:2,ubuntu:12,uint8:20,under:[5,11,18],underscor:[2,20],undershoot:6,undershoot_relative_weight_cent:6,undershoot_tau_cent:6,underspecifi:6,understand:[4,6],understood:6,uneven:3,unfinish:1,uniform:[3,6,7,10],uniformli:2,uniqu:6,unless:[3,5,9],unnam:4,unnecessari:4,unstabl:1,unsupervis:7,unsur:[1,4],until:[2,3],updat:[1,2,6],usabl:7,usag:[1,3,10,12],use:[1,2,3,4,5,6,7,9,10,15,16,17,20],used:[2,3,4,5,6,7,9,10,15,17,20],useful:2,user:[2,5,12],user_paramet:[2,20],user_parameter:2,uses:[2,4,7,10],using:[2,4,5,6,7,9,11,12,15,18,20],usual:[2,20],util:[1,12],utilitari:12,v_0:6,v_n:[6,10],valid:[1,7],valu:[1,2,3,5,6,7,10,17,20],value_from_config:2,value_to_config:2,variabl:[1,3,4,5,6,10,11,12,15,18,20],variable_describ:12,variabledelai:3,variat:[7,15],vdim:7,veri:[2,7,9,10,18,20],version:[3,6,7,10,12,13],via:[2,6,13],video:[2,9,15,19,20],videoread:9,view:2,virtual:[4,6],virtualparamet:[2,6],virtualretina:[2,4,9,12,17],vision:12,visual:[0,2,7,9,12,20],volatil:2,volum:2,wai:[1,2,4,7,17,20],walk:7,want:[2,3,7,9,11,15,16,18,19],warn:[1,2],wave:0,weight:[2,3,4,6,9,10,11,15,18,20],well:[6,9,10,15],were:1,what:[11,17,18],when:[1,2,3,4,5,6,7,9,10,12,20],where:[1,2,6,7,15],whether:[1,2,3,4,6,7,9,10,20],which:[0,1,2,3,4,5,6,9,10,14,17,20],whole:[6,16],why:20,wide:10,width:[2,3,10],wikipedia:10,wildcard:7,window:2,window_length:2,wise:3,within:5,without:[1,2,3,7,10,20],wohrer200901:4,wohrer200923:6,wohrer2009:6,wohrer:[4,6,17],work:[2,5,9,11,15,17,18,20],would:[0,12],wrap:[1,2,11,18],write:6,write_json:6,write_xml:6,www:10,x__deg:6,x_n:[6,10],x_pad:[11,18],xlabel:[10,15],xlim:[2,7],xml:[6,9],y__deg:6,y_n:[6,10],ylabel:[10,15],ylim:[2,10],you:[0,1,2,3,4,5,6,7,9,12,14,15,16,17,18,19,20],your:[2,3,9,10,12,13,15,16,18,19,20],yourself:20,zero:[3,7,11,18,19],zero_grad:20},titles:["Build your own Model","Changelog","The API: Convis classes and modules","Filters <cite>convis.filters</cite>","Models in <cite>convis.models</cite>","Optimizers <cite>convis.optimizer</cite>","VirtualRetina-like Simulator","Streams <cite>convis.streams</cite>","Automatic Tests <cite>convis.tests</cite>","Examples","Filters and Layers","Extending Conv3d","Welcome to the documentation of Convis","Installation","Models","A Convolutional Retina Model","Linear-Nonlinear Models","Retina Model","PyTorch Basics","Fitting Data","Usage"],titleterms:{"class":2,"export":20,"import":20,"new":0,"switch":20,The:2,Use:15,Using:20,all:[4,20],api:2,autograd:18,autom:18,automat:8,base:2,basic:18,between:20,bug:12,build:[0,15,18],changelog:1,combin:4,comput:[18,20],configur:[17,20],content:2,contribut:12,conv3d:[11,18],convi:[2,3,4,5,6,7,8,9,12,18],convolut:[4,10,15],cpu:20,current:1,data:[2,9,19],defin:0,describ:2,differenti:18,directli:17,disabl:20,document:12,enabl:20,exampl:9,extend:[11,18],extens:18,field:10,file:17,filter:[3,6,10,17],find:4,fit:[9,19],found:12,gener:[6,9,10,17,19],github:1,global:20,gpu:20,graph:20,hand:20,indic:12,input:20,instal:[12,13],kernel:3,layer:[0,4,10,18],like:6,linear:[4,16],list:20,method:2,model:[0,4,9,14,15,16,17,18,20],modul:2,nonlinear:[4,10,16],object:[2,20],one:4,optim:[5,20],orient:0,output:[18,20],overview:[6,17],own:0,paramet:20,pytorch:18,quickstart:9,recept:10,recurs:10,refer:6,retina:[6,15,17],retina_virtualretina:6,run:[9,20],runner:20,sampl:2,select:0,simul:[6,9],spike:[9,10,19],stream:7,submodul:4,tabl:2,tempor:10,tensor:18,test:8,torch:18,usag:[16,20],util:2,utilitari:2,variabl:2,variable_describ:2,version:1,virtualretina:6,want:12,welcom:12,xml:17,your:0,yourself:15}})