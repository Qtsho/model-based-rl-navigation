import  matplotlib.pyplot as plt
import numpy as np
import csv


plt.plot([1.0486487,0.950721,0.89624006,0.9323419,0.9605029,1.0476812,0.9843486,1.0794234,1.024664,0.9573312,1.0542804,0.954102,0.97145915,1.0022712,0.97909015,0.8922227,1.0375279,0.9822209,1.0368408,1.0717407,1.0330275,1.035847,1.0128428,1.0155925,0.9583359,1.0207125,0.97283316,1.0616628,1.0653447,0.93251437,0.92024106,1.0152582,0.9869849,0.9710393,1.1164414,1.0736486,1.0625086,0.92304516,0.94884473,1.0008702,1.0338858,0.8892937,1.0130839,0.9912489,0.90803117,0.9797271,0.917062,0.92145175,1.0150081,0.9993256,0.9028001,0.92186695,1.0005307,0.9602793,1.0024786,1.028649,1.0094603,1.1135513,0.9334851,0.93139076,0.93330055,0.8753465,0.98208255,1.005228,0.962292,0.98225003,1.0460464,0.88001424,0.9990924,0.9884112,0.9032125,0.9695311,1.0236343,0.91015464,1.0031543,0.9301443,0.94774723,1.0161389,1.0257422,0.909839,0.9161126,0.9618462,0.9675289,1.0178676,1.0317383,0.8730712,0.9611775,1.004964,1.0103315,1.009986,0.9580676,0.97596407,0.97001076,1.0018435,0.9306125,1.0508956,0.9804477,0.94666576,0.984088,0.94943136,1.1264073,0.9589946,0.9222584,0.94891816,0.8932679,0.8289942,1.1199946,1.0923716,0.86340046,1.2538639,0.90445346,0.93361217,1.0587543,0.90256447,0.9140437,0.8255732,0.8899679,0.9619698,1.030588,1.2222615,0.9734327,0.9613635,0.98913413,1.0781668,0.8240874,0.7893211,0.92291814,1.1275216,0.8871893,1.0753449,0.8317818,0.9637863,0.97078794,0.88545793,0.9818092,0.9125317,0.8754838,0.9117176,0.8563767,1.0516775,1.1989065,1.1464937,0.9741793,1.1552746,1.0110377,1.103006,1.006344,1.0769608,0.83108026,0.8814852,0.7070492,1.0158014,1.0356854,0.8110215,0.8766667,1.2481164,1.0158043,0.9146835,0.697687,1.0091847,1.1479402,1.0588387,0.98081833,1.0706927,1.0693878,1.074452,0.9180703,0.81747276,0.85906523,0.9515589,0.9003572,0.952063,0.9959271,0.9808466,1.006857,0.9206054,0.8921593,1.0513319,0.99594855,0.8451329,0.78692293,1.0590245,1.0708828,0.9639476,0.9417024,0.8728514,0.89798427,0.93283683,0.84451336,0.86653566,0.98749685,0.88140416,0.91870517,0.98755103,0.8724156,0.8548198,0.7319792,1.1083099,1.0076429,0.9973499,0.85517377,0.90435094,1.0222425,0.74987215,0.685905,1.3407868,1.1255637,0.70080394,0.9811964,0.9607373,0.9003687,0.7954127,1.1037927,0.91740036,1.2964104,0.69813377,0.96616435,1.0802321,0.8620882,0.7802181,0.9986269,0.9573636,0.79591876,1.168665,1.0564739,0.94727165,1.0372995,0.89940405,0.8310849,0.91616964,0.8858798,1.0944921,0.9755316,0.9680241,1.1011368,0.8289681,0.93517303,0.9406623,1.3184916,0.7176257,1.1203381,0.71720004,0.944066,0.8467066,0.7860239,0.79728955,0.94313097,0.90259457,0.90325856,0.73213315,0.98041254,0.58889717,0.74072,0.7284219,0.9512666,0.87810296,0.95591927,0.8807266,1.0211766,0.72211987,0.86234266,0.99458164,0.8883731,0.8815811,0.80538684,0.93624544,0.73430175,0.9311163,0.8412061,1.1153857,0.8355296,0.92612416,0.8917597,0.916199,0.871558,0.8138712,1.1371391,0.7616405,1.0420009,1.035323,0.9131453,0.8220946,0.7396181,1.0725664,1.0501989,0.83442444,0.8166886,0.92154264,1.045576,1.0297812,1.0802864,1.2321404,0.61282235,0.61343426,0.91083735,1.0930486,1.1076515,1.0330187,0.8206854,0.8539789,0.76557034,1.2179853,0.83471936,0.87826246,1.235034,0.8522412,0.812294,0.8608344,1.054935,1.02769,0.9058717,0.79306036,1.0877343,0.94520885,0.79885125,1.2348446,1.1141521,0.9995818,0.92986465,0.67503977,0.8594916,0.9858468,0.9967068,0.84164834,0.67925626,0.68971133,0.9434588,1.0583812,0.8791694,0.77805537,0.9338135,0.9833763,0.9538801,0.99803287,1.0539415,1.1605064,0.8212827,0.7044348,0.9733932,1.0159895,0.7410535,0.76654845,0.998427,0.7906639,0.75264376,0.7408648,1.036138,0.79579407,1.0432032,0.7734983,0.8849934,0.940358,0.9415657,1.0360308,0.9669902,1.0972911,0.97713304,1.1033138,0.9386185,0.8551318,0.7949348,1.0622275,0.7965668,1.0717714,0.9237349,0.97191733,1.2318646,0.81295615,0.85986924,0.728502,0.8046933,0.8987842,0.8029544,0.7402444,0.91552657,0.8702955,0.86188364,0.81729746,1.0470123,1.0006292,0.73858064,1.0244765,1.1081566,1.2329787,0.9935701,0.8325737,0.94534665,1.1921002,0.9264806,0.82467,0.96880835,0.8510695,0.9644583,0.76926357,0.79772884,0.7911199,0.90200764,0.97162557,1.0389059,0.99735427,0.7821625,1.0018225,0.88492376,1.1000618,0.6546118,0.85718036,1.0521699,1.0795783,0.84540683,0.7198732,0.9705453,0.8313217,0.8122671,1.1776648,0.76322865,0.8803673,0.9778886,1.0547731,0.8608219,0.7563958,0.9182739,0.7777799,0.725224,0.67360884,0.9761655,0.9320193,0.8638914,0.8821783,0.7858062,0.96310234,1.0525793,0.9285292,0.7912757,0.93019897,1.1001453,0.9467118,1.0183827,0.8999736,0.8329218,0.74850225,0.86364746,1.0238,0.91798526,1.0317403,0.8981061,1.1296444,0.968348,1.0712415,0.8564771,0.9896632,0.7060645,0.7745332,0.7677006,0.93011945,0.9028136,0.9162118,0.8661509,0.9708283,1.1183639,0.7515731,0.96999216,1.1302394,1.0001589,0.6051974,0.7437112,0.9327933,0.7237735,1.0112984,0.8933583,1.010805,0.83617043,0.7335768,0.87522095,0.7823116,0.76639587,1.1950923,0.85956544,0.8398502,0.8210735,0.98120093,0.75360686,1.1238434,0.85354537,0.6768258,0.94318765,0.9477501,0.79970986,0.9104705,0.65860695,1.0669082,1.0130601,0.96925956,0.7846167,1.06069,0.847186,0.87217396,0.92892146,0.74612194,0.73548484,1.0358392,0.98305815,0.88820404,0.91686386,0.9668481,0.93584925,1.2409855,1.0428559,0.98491746,1.0390368,0.8829687,0.71376944,0.8197182,0.9294932,0.9827784,0.9442585,0.8943899,1.010538,0.8133035,0.7854595,0.93479615,1.105105,1.0175382,1.1241498,0.77015895,0.92503524,0.8772238,0.97585917,0.8082641,0.93819064,0.9171486,0.6951707,0.7993683,0.9563752,0.92477465,0.9496231,0.6854612,0.9377853,0.98885936,0.71775585,1.1451125,1.0658444,0.7652533,0.779645,1.051596,0.8896881,0.5998245,0.6996873,0.92029667,0.8540763,1.0759631,0.9169953,0.85052806,0.787349,0.79459333,0.855839,0.959387,1.0459429,0.97158414,0.9044318,0.8490897,0.87457174,0.9638343,0.97687775,0.9040513,0.9449951,0.81731683,0.72458774,0.9322117,0.9138613,0.96538204,0.9002827,0.80159336,1.0509602,0.9603364,0.8410461,0.7951785,0.93854934,0.88398933,0.7626066,1.0455211,1.089923,1.0575106,0.71326566,1.0514396,0.76998925,0.7978966,0.9245615,0.73653096,0.84319,0.73222846,0.98467916,1.02959,0.7221038,0.99972135,0.690991,0.9958572,0.9743927,1.0579578,0.70611316,0.87101334,0.9256297,0.8795233,0.68742424,0.7643791,0.7863099,0.63974714,1.5012914,0.8230273,0.78593093,0.8870109,0.8781112,1.7414318,1.4088722,0.68872803,0.6240099,0.79564196,0.7879371,0.79876596,1.6534699,1.0208513,1.5980655,0.9063155,0.76182,0.60842466,0.8278444,0.8608702,0.77797866,0.7437751,1.4720439,0.7735018,0.8101337,0.7030428,0.7130813,1.57138,0.7965169,1.4264625,0.850761,0.68716174,0.987055,0.82371825,0.8156755,0.623273,1.1113312,0.7559452,0.8711622,0.8673534,0.93136024,0.87933475,0.5641884,0.826853,0.79741436,1.6096622,0.76450175,0.86681825,0.70051616,0.82822067,0.6461421,1.4999005,0.81489843,0.93349797,0.7161923,0.76663,0.67493343,0.6769552,0.8172267,0.6440366,1.0460334,0.8881175,1.4110049,0.7524717,0.7087143,0.7616287,0.72681093,0.78975016,1.1702143,0.9461255,0.69841194,1.3607253,1.6651659,0.98964167,0.7861864,0.7180922,0.70171523,0.87607616,0.6064618,0.75522953,0.8481324,1.4681631,0.7427476,0.79991156,0.6814921,0.66153795,0.85371095,0.8733731,1.0447332,1.4172548,0.73935884,0.8872876,0.6724405,0.7236154,0.8797942,0.7830809,0.6643676,0.8893629,1.0966231,0.81598645,0.94113964,0.8136687,0.642085,0.8028231,0.6283162,0.6308133,1.6922398,0.86318463,0.68026716,0.8258907,1.5969967,0.7692535,0.90380794,0.79164714,1.675774,0.71428555,0.7704916,1.7635981,0.7326818,0.7490662,0.6179066,0.74630475,0.93352467,0.87454146,0.61097497,0.7848151,0.91633457,0.843617,0.8591438,1.1255862,0.7969335,0.6454451,0.6772632,0.9301524,0.8009146,0.8705356,0.8546829,1.7717277,0.8379652,0.85860586,0.73130375,0.7082679,0.85363746,0.8630099,0.76703614,0.79063576,0.7316901,0.55597025,0.62366194,0.696111,0.65235835,0.58600587,0.8029191,0.5794186,0.6258947,0.7076276,0.84016484,0.93942255,0.89750385,0.9523671,0.83698225,0.96664697,1.6762381,0.7961716,0.7318252,1.7722751,0.8412285,0.9133222,0.74518484,1.7458161,0.6603933,0.7663558,0.8159795,0.9060437,0.7235339,1.4624113,0.7460844,0.7151883,0.72984666,0.876153,0.67779607,0.92602086,0.69237655,0.59528977,0.6809156,0.77466255,0.7871477,0.7266751,0.5503505,0.7872763,0.61790323,1.7446694,0.78192884,0.7964745,0.832979,0.77603745,0.7830694,0.93984157,1.2266668,0.8726484,1.63162,0.7650091,0.5351465,0.91259074,0.8938828,0.858578,0.81171227,0.79466915,0.74072266,0.7592952,0.92069846,0.8452661,0.6284533,0.6858656,0.8457009,0.7105331,1.8066741,1.6548325,0.7984824,1.0036991,0.63693434,0.871206,0.8297141,0.77754337,1.1147543,0.6194227,0.82332164,0.7487982,0.93569154,0.7607593,0.7654724,0.698358,0.7674958,1.0103089,0.8304474,0.8936,0.6215202,1.1154925,0.7150293,0.9277745,0.8626942,0.6085944,0.72723013,0.6408834,1.0637907,0.94285184,0.77512646,1.5580372,0.72377706,0.7524772,1.7432485,0.7750061,1.7431479,1.086737,0.79227066,0.640574,0.67063046,0.7550146,0.8255225,0.79704636,0.72777534,0.8215403,0.8141887,0.63395375,1.763466,0.5491604,0.7398014,0.68622637,0.8353352,0.75358576,0.9167814,0.78246474,0.6242108,0.80382806,0.5685136,0.7083513,0.80395913,0.80198234,0.68358964,0.6617603,0.741164,0.8186288,0.7650347,0.9191149,0.8548749,1.0731922,0.8625848,1.7118855,0.8269801,0.8224352,0.89325494,0.63079995,0.8959281,0.78797287,0.4895029,0.48380348,0.7560081,0.6937235,1.4077525,0.58728045,0.5315725,1.3946533,0.6353565,0.6476224,0.5663368,0.62294406,0.7233197,0.49631277,0.60678315,0.63720196,0.7366063,0.53718287,0.63153434,0.5351803,0.5964286,0.6877866,2.271136,0.6297426,2.2347643,0.7320551,0.69007987,0.5182176,0.5544276,0.73845696,0.5884991,0.5782024,0.64855766,0.5735414,0.6630721,0.65699214,0.4545246,0.63584137,0.67047614,0.63895625,0.5572809,0.6595475,0.8232781,0.6900098,0.6248472,0.71266156,0.41456345,0.6134333,0.5258982,0.7045674,0.5833914,1.5470033,1.3754317,0.63522243,0.43854034,0.5220935,0.7208133,2.0261173,0.74765426,2.0048664,0.64770454,0.76577836,0.6405926,0.5587401,0.6896844,0.54061145,0.56853,0.5143642,0.7500706,0.7092436,1.2967438,0.6281298,0.49577108,0.72868925,1.4596715,0.45068058,0.60976905,2.4196508,0.56207156,0.63365173,0.5253125,0.71990424,0.6190895,0.59112483,0.73329514,0.7469921,0.7267859,0.69642705,0.5049783,3.1278813,0.615584,0.715357,0.69876146,0.8563058,0.5012223,0.82562566,1.433248,0.6146752,1.609445,0.53178924,0.60061014,0.6407697,0.8005441,0.70023507,0.6812127,0.75665784,0.5710394,0.63586307,2.2330568,0.6025892,2.2835073,0.525876,0.59455085,0.6453328,1.3832365,0.53369385,0.7222802,0.569527,1.4297705,0.6665833,2.2599738,1.9757004,0.60524315,1.6975385,0.6642673,0.6318285,0.55950737,2.219824,2.1349583,2.1941876,0.5984747,1.3978709,0.63258344,0.6213434,1.4927373,1.39635,2.3299985,0.7096049,0.6901695,0.67611784,0.627275,0.5107462,0.6142143,0.5961157,2.4367812,0.6031657,0.464366,2.2626777,0.59150267,0.56974435,0.7198059,0.6610624,0.79394716,0.6186467,0.67478657,0.69223017,0.56486136,1.4510897,1.6802698,0.65342003,0.891778,0.5351725,2.2501097,0.6771641,0.64548755,0.51295114,1.5582324,0.81321067,0.66441035,0.5342867,1.6790184,0.66060925,0.5401533,0.48229003,0.68724394,0.6615771,0.6217428,0.5855903,0.61751163,0.57618964,1.375135,0.6020997,0.5834363,0.63652474,2.2571723,0.7023911,0.5471882,2.1294127,0.66762847,0.5124532,0.7990349,0.43796816,0.6320648,0.39981428,0.5375815,0.64919835,0.5053374,1.4794035,0.5483855,1.5071586,0.46850023,0.5458153,0.5654268,1.3544961,0.4937221,0.55231804,0.5835006,0.60754746,0.575175,0.72568196,0.5116102,1.2626299,2.0234082,0.4510099,0.50183004,0.59802,1.1840652,0.5145542,0.50132376,0.62213624,0.49798954,3.0738304,0.6459494,0.5363781,1.3410907,0.42160407,2.0572445,0.5350905,1.3326439,2.4017024,0.5786329,0.67000824,0.60385734,0.4835366,0.47318792,0.5850462,0.5209456,0.51612616,1.2487727,0.70302486,0.72184855,0.526061,0.5203152,0.5383261,0.6331279,0.5933752,0.62990254,0.4856422,0.41370264,0.64447886,0.79032403,0.7168073,0.5180885,1.5523213,0.5781093,0.50483245,3.0907574,0.47506008,0.5751422,0.5933257,0.70291567,0.41789773,0.5620781,0.51187915,0.5951869,0.50169533,0.50678486,2.2512918,0.52253014,0.55803865,0.46887562,0.51489526,0.5136702,2.6737459,0.90110826,2.1904933,0.6577473,1.5780879,2.081848,0.7191286,0.6619171,0.48665643,2.0943205,1.5089918,0.42108154,0.51228386,2.0553553,0.88907504,2.919811,0.5700199,2.0082383,0.5487197,0.50902396,1.4390597,0.47707924,1.9325784,0.6986535,0.671025,0.5697916,0.8994844,0.8047413,0.8138132,1.5477533,0.45904362,0.5727601,0.5765129,2.0261657,0.7724536,0.51976633,0.5569054,1.2849149,0.5599088,0.45887432,0.68526983,1.7813407,2.1895611,0.5491763,0.454366,2.1661747,0.70859593,0.5667419,0.6161073,0.87493414,0.62624496,0.5485106,0.5451072,0.6630954,0.61190516,0.5057046,0.61763555,1.2669097,0.9402321,0.66099596,0.57812196,0.61892897,1.3013381,1.5750097,0.6362711,1.5364813,1.5922195,0.6508979,0.6202618,0.6056943,0.41315022,0.5746352,1.3750148,2.8992186,2.2682893,0.5604369,1.275452,0.50986356,0.5952301,0.49304095,0.6285267,2.2426825,0.7456813,0.7380139,1.5331612,0.5141898,0.55941063,1.236131,0.48968312,0.533837,0.55018693,1.2401146,0.52469087,0.46880016,0.49475765,0.5271365,0.5730346,1.3602028,0.5266004,0.6380692,0.5138354,1.3836507,1.0021259,2.85841,0.5518509,0.65391904,0.5068273,0.6323554,0.5838701,0.6208419,0.612916,1.5381943,0.53360015,0.4834559,1.3771788,0.5412356,1.2844357,0.47368822,0.53269833,1.4230088,0.48292407,0.51680154,0.5880132,0.6629799,0.69020844,2.0494132,0.8459468,0.53271675,0.9738345,0.40511164,1.1133678,1.347617,0.5914209,0.3909731,1.0161686,0.61936975,0.341897,1.43157,1.3037263,0.7213685,0.46140316,0.34286365,1.3047832,0.3989211,1.6409415,0.3171564,0.52198696,0.43507442,0.9747963,0.567383,0.8008608,0.28923014,0.44135574,0.4292736,1.2107428,1.4530059,0.5081636,0.43246874,1.996439,1.5714403,2.1064692,0.3479686,0.6270788,0.53008395,1.725665,1.0712987,0.39502284,0.41612864,0.4246029,0.49315128,1.3182355,0.75687414,0.5163315,0.34608975,1.3501424,0.658469,1.9641804,0.7933552,0.49692154,0.5462846,2.0441828,0.70154804,0.5091812,1.2719123,0.5145002,1.3913898,1.2396598,1.6102663,0.46196494,1.2754096,0.82437754,0.5789313,1.2462622,1.1714355,1.5780452,1.3523964,1.425171,2.0029876,0.46218896,2.2688034,0.59672,0.46063423,2.2103646,1.3266472,0.55186963,0.5017588,1.2625719,0.9900661,0.49318838,1.2805192,0.7508934,1.1603868,1.1580626,0.44545123,0.4733173,0.44304135,0.4089081,0.46059123,0.5012643,1.0036563,2.1238291,0.53798777,0.42068967,0.6886801,1.1427383,1.3284292,1.3600364,0.7096498,0.38996232,1.4038473,0.38204813,1.2242175,0.39100668,0.57412004,1.3220774,1.1319358,0.5887504,0.44413686,1.3565747,0.46446046,0.48636222,1.8566443,0.53944,1.8678545,0.42819408,0.41014478,1.228653,1.7320434,1.0673413,0.42510244,0.44341794,0.59060174,1.0456368,1.2058727,1.0703475,1.3980994,1.2234814,0.50707155,2.1418626,2.268252,0.5472596,0.4311084,2.2113872,0.41816422,0.46741197,0.43366185,1.3597342,0.34814915,0.42820027,1.4435514,0.44875216,0.95660883,1.1947237,0.5721479,1.1900018,1.3708643,0.5118678,0.48046827,0.79354745,0.4993662,0.52250695,0.43936566,1.3048717,0.8072691,2.6881974,1.1160785,0.54905957,0.51993626,0.4559532,0.5085153,0.46593297,0.4271419,2.1986516,0.5558606,1.3790079,0.49396157,0.48986602,0.5140713,0.50389326,0.5974421,0.43039677,0.51223063,0.7521226,1.294094,0.41806474,0.3643982,0.35699806,0.96347,0.45800996,0.45337844,0.81302404,0.4879147,2.0697653,1.2749201,0.8299787,0.8437678,1.0235541,0.47727323,2.376796,1.3101147,1.6205312,0.45031548,0.4561409,0.39858648,1.190969,0.5030656,0.412504,2.192209,1.5243431,0.88535184,0.5318964,0.42140666,0.40766189,1.3283556,0.5851453,1.056468,0.45559072,0.4258943,0.48381114,0.2904015,0.45881358,0.54245657,2.5582228,1.3713737,1.2080288,0.443089,0.46671984,0.97495604,0.48134637,0.50663066,0.5333943,0.6040415,0.38101998,0.56716585,1.2888535,1.3681086,0.68751264,1.3372021,2.400659,0.63538814,0.56423014,0.57847804,0.60452825,0.45835486,0.49678364,0.4305571,1.540979,2.0445254,1.4224758,0.85191625,1.2947141,0.45966473,1.1126168,0.3225761,1.1911358,0.5715236,0.61478555,0.42625007,1.4791046,1.2908024,1.430706,0.59309816,0.5521949,1.2053984,0.5276455,0.4174888,2.600166,0.44725552,0.5564449,0.68324965,0.48549286,1.323114,0.47791934,0.5041789,0.58369046,1.4400877,1.034386,1.2895662,0.40481964,0.4222783,0.40642366,0.79905564,1.3383113,1.4034791,1.4673308,1.9397874,1.3510795,0.3911313,0.3573494,0.44598484,0.36586776,0.44000962,0.42977247,2.0182898,0.49407753,1.8911761,0.40990695,0.51776606,1.1574397,1.8555015,0.41209492,0.51087904,1.4728976,0.4465752,1.3932763,1.182606,0.31153205,0.5015505,0.41102913,0.65444285,1.1649158,1.7209864,1.1389211,0.99520165,1.6187402,1.1090759,0.46584913,0.48890588,0.6839376,0.43186697,0.40397045,1.0842767,0.9452993,1.7626929,1.9732116,0.5662415,2.1554844,0.43937254,1.3504934,1.4818692,1.4687439,0.50666183,0.38070905,0.4263595,0.48716235,0.46514025,1.5003101,1.0558971,0.4808723,1.6087974,1.5243827,0.84084195,0.6048812,0.48275912,1.4484881,0.5597626,0.71170276,1.5089067,0.5115248,1.4071689,0.6943538,0.45796525,0.4955798,0.6183227,0.5587629,0.50524026,0.49621657,0.8544485,0.3526887,0.45525873,1.3992923,0.49551305,2.2739012,0.4935219,1.4013515,1.5697055,0.41467834,0.8468933,0.40501547,0.9491202,0.4494337,1.7746053,0.471207,1.0406474,0.51004404,0.36623046,1.4397421,1.1291375,1.4118886,0.4828339,0.46490404,0.46417856,0.57603556,0.5433407,0.40021887,0.45364484,0.63764495,1.4913672,1.7817444,0.5430722,0.5316112,1.3198887,1.2788743,0.450045,1.2367333,1.3907475,0.5098527,1.3821298,0.43006644,0.6143353,1.579794,0.52083915,0.49486575,1.164781,0.5507049,0.46598554,2.5300276,0.5290515,0.58761543,1.4055132,1.3950554,1.7970605,0.5862195,0.89687854,0.5045102,0.6581487,0.7262962,1.0892981,1.4828233,1.2323768,0.8535239,2.075204,0.49300504,1.4145155,1.2403892,1.5730613,1.1272423,0.40748724,0.5127898,0.70804554,0.35948098,0.4711143,0.39615443,0.49104443,0.49676478,1.4540415,0.50565416,1.0946388,0.82526445,0.36530462,1.6421419,1.5804936,0.42304695,1.1087418,1.4560156,0.52498376,0.47167218,0.5314228,2.2946951,0.53431094,0.98895675,1.5399736,0.43807283,1.5613059,0.5300481,0.49372044,1.4751321,1.5657307,0.49485764,0.49166617,0.53536826,0.57418066,0.48389244,0.4547005,1.3150281,1.4250702,0.45253357,0.5239914,1.4624032,1.1220908,0.475674,0.5156662,1.4542102,0.3786842,0.61306244,0.5529478,0.44298592,1.3478206,1.6987748,0.39113092,0.51412827,0.46225402,0.8151118,0.43830094,1.4620775,0.632911,0.36926875,0.48389462,0.5204846,1.1018208,1.5357174,1.323349,0.49718904,1.400092,1.0297312,1.2652466,0.6081628,1.6073633,1.6576914,1.5432738,2.3438394,1.3244091,1.3354913,0.55009276,1.9218282,1.4063202,0.519617,0.5564325,1.5220151,0.48089984,0.5161764,0.4817432,0.52933174,0.58579445,0.35940695,0.48267594,0.5652892,0.827199,0.45550248,2.5000403,0.56197095,0.4767829,1.3826542,0.47560945,0.43708608,0.54096526,0.39073166,0.54518,0.43204752,0.4648033,0.43089473,0.4919397,0.7978588,2.4818609,0.46596804,1.5516315,0.9389337,0.36470318,0.62964845,0.44068578,0.5267313,1.4621433,0.462226,0.49191007,0.5757255,1.1660442,0.6680998,0.43888855,0.5555597,0.5631754,1.3120188,0.55370814,1.3745557,0.6186053,0.48931086,0.41884187,0.5638385,0.7065098,0.9284881,0.54658717,0.48480964,0.45452252,1.5682052,1.3229449,0.46245766,0.5053789,1.5308971,1.5243425,0.40158084,0.43953776,0.47054186,1.5276455,0.40682778,0.5249481,0.33910632,1.4158436,0.51858383,0.42464018,0.5130949,0.42297196,0.47962025,2.2371361,1.8466195,0.4973662,0.6137714,0.41266763,0.4429523,2.0632184,0.43754652,0.4890002,1.0655974,0.46681222,1.7093283,0.4238188,0.48271897,0.64304477,1.4526467,0.5095261,0.4616847,1.3547134,0.46193945,0.6411447,2.3317533,1.540706,0.5167738,1.2624503,0.53491396,1.2452788,1.364356,0.54832345,0.5310157,2.0581498,0.5952477,0.58293724,0.56770587,1.2692916,0.48401546,2.392648,0.5341206,0.5331994,1.9241142,0.38903594,0.47066274,0.5897078,0.75089926,0.5898028,0.54408824,0.5193169,1.5581411,0.44261363,0.5246957,1.1199027,1.4234222,1.6516839,0.45390955,1.5694294,0.49386707,1.3088707,0.6043461,0.5465891,0.4539533,0.49081603,2.055406,0.47306085,1.4176702,0.3745631,1.1477259,0.4491972,0.89281535,1.6123219,0.5719106,0.81702346,0.4686408,0.4630324,0.7099278,0.757746,0.6019402,0.42126513,0.61842376,2.1324315,1.5381838,0.7366944,1.131367,0.45965472,1.4437313,0.44706842,0.3973931,0.5176758,1.6509622,0.3873714,0.5013199,1.3469863,0.52392256,1.8325624,0.5396867,0.40162554,0.4326787,1.5120913,0.48332453,0.5877215,0.47443298,0.58972025,0.74962026,0.41071174,0.7004626,1.4784145,0.5833191,1.3316604,0.56958884,1.8592429,1.8903233,0.7360518,0.4728868,2.5494678,1.520628,0.7495496,0.4061021,0.53167254,0.51620734,0.3542401,1.5909977,1.1063474,0.3866481,1.4819889,1.2776976,0.54751325,0.6007031
])
plt.xlabel('Trainning step')
plt.ylabel('Loss')
plt.show()