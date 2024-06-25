# Sentiment Analysis

### Sentiment Analysis of Tweets


---

sentiment analysis using HuggingFace transformers in Python.

---

Required Packages: transformers, torch, numpy, pandas

```python3
pip3 install transformers torch numpy pandas
```


Model: https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english

dataset: https://www.kaggle.com/datasets/yessicatuteja/sentiment-analysis-of-tweets

---


```bash
(p3) lynk@APPLEs-MacBook-Pro Desktop % python -u "/Users/lynk/Desktop/sentiment-analysis.py"
2024-06-25 21:14:23.221164: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
/opt/miniconda3/envs/p3/lib/python3.11/site-packages/threadpoolctl.py:1214: RuntimeWarning: 
Found Intel OpenMP ('libiomp') and LLVM OpenMP ('libomp') loaded at
the same time. Both libraries are known to be incompatible and this
can cause random crashes or deadlocks on Linux when loaded in the
same Python program.
Using threadpoolctl may cause crashes or deadlocks. For more
information and possible workarounds, please see
    https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md

  warnings.warn(msg, RuntimeWarning)
         tweet_id  sentiment         author                                            content
9355   1962529654       hate    beingnobody  Oh, fuck me. I've just returned from the Super...
4886   1960544150      worry        KevMain  Moving office tomorrow after 3 years at this o...
11342  1963187865    neutral       djjeddyb  #thingsmummysaid...rollercoasters are only mul...
13479  1964048928   surprise      SamLuck19  Killed a pigeon today  Thought it was going to...
34118  1752796171      empty          zenbb  @gk2007 Yu th? trang nï¿½y xem  http://bit.ly/...
...           ...        ...            ...                                                ...
35711  1753158239  happiness      mini_ritz  @Jae878  thanks  I'm holding my mini laser lig...
19617  1966307267    neutral  NathalieCaron  @Patti0713 It's not on my cable carrier. Only ...
25652  1695163864      worry   VioletAngel1  is cheering on the Arsenal Ladies!!    and mis...
12777  1963724755       hate      revmink33  The only thing about preparing 5 sermons in on...
28025  1696006038    sadness      spryfaery  @Heart_song Glad you had a wondrous Beltaine! ...

[200 rows x 4 columns]
[{'label': 'NEGATIVE', 'score': 0.9991938471794128}, {'label': 'NEGATIVE', 'score': 0.8212727308273315}, {'label': 'NEGATIVE', 'score': 0.9679269790649414}, {'label': 'NEGATIVE', 'score': 0.9984161853790283}, {'label': 'NEGATIVE', 'score': 0.851008951663971}, {'label': 'NEGATIVE', 'score': 0.7902448773384094}, {'label': 'NEGATIVE', 'score': 0.9991554021835327}, {'label': 'NEGATIVE', 'score': 0.897927463054657}, {'label': 'NEGATIVE', 'score': 0.9929895401000977}, {'label': 'POSITIVE', 'score': 0.999870777130127}, {'label': 'NEGATIVE', 'score': 0.9920926690101624}, {'label': 'POSITIVE', 'score': 0.9535270929336548}, {'label': 'NEGATIVE', 'score': 0.9953281879425049}, {'label': 'NEGATIVE', 'score': 0.9991187453269958}, {'label': 'POSITIVE', 'score': 0.9995375871658325}, {'label': 'POSITIVE', 'score': 0.9633103013038635}, {'label': 'POSITIVE', 'score': 0.9995717406272888}, {'label': 'NEGATIVE', 'score': 0.9942452311515808}, {'label': 'NEGATIVE', 'score': 0.6153771877288818}, {'label': 'NEGATIVE', 'score': 0.9924334287643433}, {'label': 'POSITIVE', 'score': 0.9462117552757263}, {'label': 'POSITIVE', 'score': 0.9957706332206726}, {'label': 'NEGATIVE', 'score': 0.9997096657752991}, {'label': 'POSITIVE', 'score': 0.9996337890625}, {'label': 'POSITIVE', 'score': 0.9785970449447632}, {'label': 'POSITIVE', 'score': 0.9965008497238159}, {'label': 'NEGATIVE', 'score': 0.97406005859375}, {'label': 'NEGATIVE', 'score': 0.9680083990097046}, {'label': 'POSITIVE', 'score': 0.9996532201766968}, {'label': 'POSITIVE', 'score': 0.8575786352157593}, {'label': 'NEGATIVE', 'score': 0.9990264177322388}, {'label': 'POSITIVE', 'score': 0.9726644158363342}, {'label': 'NEGATIVE', 'score': 0.9972148537635803}, {'label': 'POSITIVE', 'score': 0.9844653010368347}, {'label': 'POSITIVE', 'score': 0.8398813605308533}, {'label': 'NEGATIVE', 'score': 0.9997662901878357}, {'label': 'NEGATIVE', 'score': 0.9996846914291382}, {'label': 'POSITIVE', 'score': 0.9348998665809631}, {'label': 'NEGATIVE', 'score': 0.9954380393028259}, {'label': 'POSITIVE', 'score': 0.998193085193634}, {'label': 'NEGATIVE', 'score': 0.9846375584602356}, {'label': 'NEGATIVE', 'score': 0.9986984729766846}, {'label': 'POSITIVE', 'score': 0.9997170567512512}, {'label': 'NEGATIVE', 'score': 0.9974105954170227}, {'label': 'POSITIVE', 'score': 0.9997467398643494}, {'label': 'POSITIVE', 'score': 0.9998138546943665}, {'label': 'NEGATIVE', 'score': 0.9981138706207275}, {'label': 'NEGATIVE', 'score': 0.9971684813499451}, {'label': 'NEGATIVE', 'score': 0.8268777132034302}, {'label': 'POSITIVE', 'score': 0.8898416757583618}, {'label': 'POSITIVE', 'score': 0.9997978806495667}, {'label': 'NEGATIVE', 'score': 0.9987711310386658}, {'label': 'NEGATIVE', 'score': 0.9962769150733948}, {'label': 'NEGATIVE', 'score': 0.9958756566047668}, {'label': 'NEGATIVE', 'score': 0.9978819489479065}, {'label': 'NEGATIVE', 'score': 0.999014139175415}, {'label': 'NEGATIVE', 'score': 0.9339135885238647}, {'label': 'NEGATIVE', 'score': 0.9943027496337891}, {'label': 'NEGATIVE', 'score': 0.9985378980636597}, {'label': 'NEGATIVE', 'score': 0.975536048412323}, {'label': 'POSITIVE', 'score': 0.9933941960334778}, {'label': 'NEGATIVE', 'score': 0.9984363913536072}, {'label': 'POSITIVE', 'score': 0.9997119307518005}, {'label': 'POSITIVE', 'score': 0.996401309967041}, {'label': 'NEGATIVE', 'score': 0.998042106628418}, {'label': 'NEGATIVE', 'score': 0.9936827421188354}, {'label': 'POSITIVE', 'score': 0.9427001476287842}, {'label': 'NEGATIVE', 'score': 0.9996917247772217}, {'label': 'POSITIVE', 'score': 0.9980125427246094}, {'label': 'NEGATIVE', 'score': 0.9982934594154358}, {'label': 'POSITIVE', 'score': 0.9991291165351868}, {'label': 'NEGATIVE', 'score': 0.999555766582489}, {'label': 'NEGATIVE', 'score': 0.9117883443832397}, {'label': 'POSITIVE', 'score': 0.9996329545974731}, {'label': 'NEGATIVE', 'score': 0.9992640614509583}, {'label': 'POSITIVE', 'score': 0.9994450211524963}, {'label': 'NEGATIVE', 'score': 0.9962683320045471}, {'label': 'NEGATIVE', 'score': 0.999613344669342}, {'label': 'POSITIVE', 'score': 0.9995705485343933}, {'label': 'POSITIVE', 'score': 0.6506785750389099}, {'label': 'NEGATIVE', 'score': 0.9989368319511414}, {'label': 'NEGATIVE', 'score': 0.7903785109519958}, {'label': 'POSITIVE', 'score': 0.9975118637084961}, {'label': 'NEGATIVE', 'score': 0.9991369843482971}, {'label': 'NEGATIVE', 'score': 0.9908849596977234}, {'label': 'NEGATIVE', 'score': 0.8972830772399902}, {'label': 'POSITIVE', 'score': 0.9835255146026611}, {'label': 'POSITIVE', 'score': 0.9998327493667603}, {'label': 'POSITIVE', 'score': 0.9996351003646851}, {'label': 'POSITIVE', 'score': 0.972537100315094}, {'label': 'POSITIVE', 'score': 0.9573084712028503}, {'label': 'POSITIVE', 'score': 0.9997953772544861}, {'label': 'POSITIVE', 'score': 0.9649789333343506}, {'label': 'NEGATIVE', 'score': 0.9988241791725159}, {'label': 'POSITIVE', 'score': 0.9991419315338135}, {'label': 'NEGATIVE', 'score': 0.9989161491394043}, {'label': 'NEGATIVE', 'score': 0.9993619322776794}, {'label': 'NEGATIVE', 'score': 0.9822791814804077}, {'label': 'POSITIVE', 'score': 0.9998759031295776}, {'label': 'NEGATIVE', 'score': 0.9991922974586487}, {'label': 'NEGATIVE', 'score': 0.9807064533233643}, {'label': 'NEGATIVE', 'score': 0.7379233837127686}, {'label': 'NEGATIVE', 'score': 0.9959179759025574}, {'label': 'NEGATIVE', 'score': 0.9805828928947449}, {'label': 'NEGATIVE', 'score': 0.9890885353088379}, {'label': 'POSITIVE', 'score': 0.980602502822876}, {'label': 'NEGATIVE', 'score': 0.9967902302742004}, {'label': 'POSITIVE', 'score': 0.903872549533844}, {'label': 'NEGATIVE', 'score': 0.9951327443122864}, {'label': 'NEGATIVE', 'score': 0.995582640171051}, {'label': 'POSITIVE', 'score': 0.9977433681488037}, {'label': 'NEGATIVE', 'score': 0.8919761776924133}, {'label': 'NEGATIVE', 'score': 0.9932039976119995}, {'label': 'POSITIVE', 'score': 0.9988875985145569}, {'label': 'POSITIVE', 'score': 0.9998418092727661}, {'label': 'NEGATIVE', 'score': 0.9492014050483704}, {'label': 'NEGATIVE', 'score': 0.9946106672286987}, {'label': 'NEGATIVE', 'score': 0.8950132131576538}, {'label': 'NEGATIVE', 'score': 0.9602644443511963}, {'label': 'POSITIVE', 'score': 0.8336657285690308}, {'label': 'POSITIVE', 'score': 0.9995779395103455}, {'label': 'NEGATIVE', 'score': 0.9358395934104919}, {'label': 'POSITIVE', 'score': 0.9998352527618408}, {'label': 'POSITIVE', 'score': 0.9176763296127319}, {'label': 'NEGATIVE', 'score': 0.999453604221344}, {'label': 'POSITIVE', 'score': 0.9857500791549683}, {'label': 'POSITIVE', 'score': 0.9903451800346375}, {'label': 'POSITIVE', 'score': 0.9996941089630127}, {'label': 'NEGATIVE', 'score': 0.9973114728927612}, {'label': 'POSITIVE', 'score': 0.994162380695343}, {'label': 'NEGATIVE', 'score': 0.9983218312263489}, {'label': 'NEGATIVE', 'score': 0.9990423321723938}, {'label': 'NEGATIVE', 'score': 0.9979262351989746}, {'label': 'NEGATIVE', 'score': 0.9953429698944092}, {'label': 'NEGATIVE', 'score': 0.9991901516914368}, {'label': 'NEGATIVE', 'score': 0.9976270794868469}, {'label': 'POSITIVE', 'score': 0.9993953704833984}, {'label': 'NEGATIVE', 'score': 0.9906971454620361}, {'label': 'NEGATIVE', 'score': 0.9825665354728699}, {'label': 'POSITIVE', 'score': 0.9831072092056274}, {'label': 'POSITIVE', 'score': 0.9589449763298035}, {'label': 'NEGATIVE', 'score': 0.9782748222351074}, {'label': 'NEGATIVE', 'score': 0.9980385899543762}, {'label': 'NEGATIVE', 'score': 0.9677836298942566}, {'label': 'POSITIVE', 'score': 0.9936049580574036}, {'label': 'NEGATIVE', 'score': 0.6863706707954407}, {'label': 'NEGATIVE', 'score': 0.985194981098175}, {'label': 'NEGATIVE', 'score': 0.9861451387405396}, {'label': 'NEGATIVE', 'score': 0.9929673671722412}, {'label': 'NEGATIVE', 'score': 0.8717076778411865}, {'label': 'NEGATIVE', 'score': 0.9986456036567688}, {'label': 'POSITIVE', 'score': 0.9912693500518799}, {'label': 'NEGATIVE', 'score': 0.9971727132797241}, {'label': 'NEGATIVE', 'score': 0.8640374541282654}, {'label': 'POSITIVE', 'score': 0.9965356588363647}, {'label': 'POSITIVE', 'score': 0.9996576309204102}, {'label': 'NEGATIVE', 'score': 0.9978697299957275}, {'label': 'NEGATIVE', 'score': 0.9906909465789795}, {'label': 'POSITIVE', 'score': 0.994864284992218}, {'label': 'POSITIVE', 'score': 0.9955570101737976}, {'label': 'NEGATIVE', 'score': 0.9880920648574829}, {'label': 'POSITIVE', 'score': 0.9998648166656494}, {'label': 'NEGATIVE', 'score': 0.6356391310691833}, {'label': 'NEGATIVE', 'score': 0.9892799258232117}, {'label': 'POSITIVE', 'score': 0.996052086353302}, {'label': 'NEGATIVE', 'score': 0.9968383312225342}, {'label': 'POSITIVE', 'score': 0.999839186668396}, {'label': 'NEGATIVE', 'score': 0.9992460012435913}, {'label': 'NEGATIVE', 'score': 0.9992926120758057}, {'label': 'NEGATIVE', 'score': 0.9726062417030334}, {'label': 'POSITIVE', 'score': 0.9289051294326782}, {'label': 'POSITIVE', 'score': 0.9983618855476379}, {'label': 'POSITIVE', 'score': 0.9993817806243896}, {'label': 'POSITIVE', 'score': 0.9995130300521851}, {'label': 'POSITIVE', 'score': 0.9998675584793091}, {'label': 'POSITIVE', 'score': 0.8328090906143188}, {'label': 'NEGATIVE', 'score': 0.9996616840362549}, {'label': 'NEGATIVE', 'score': 0.9979817867279053}, {'label': 'NEGATIVE', 'score': 0.9996119141578674}, {'label': 'NEGATIVE', 'score': 0.9946340322494507}, {'label': 'NEGATIVE', 'score': 0.5661691427230835}, {'label': 'NEGATIVE', 'score': 0.9498242139816284}, {'label': 'NEGATIVE', 'score': 0.952165424823761}, {'label': 'NEGATIVE', 'score': 0.9660447239875793}, {'label': 'POSITIVE', 'score': 0.831809401512146}, {'label': 'NEGATIVE', 'score': 0.9989569187164307}, {'label': 'NEGATIVE', 'score': 0.9828109741210938}, {'label': 'NEGATIVE', 'score': 0.998397171497345}, {'label': 'NEGATIVE', 'score': 0.9981616139411926}, {'label': 'NEGATIVE', 'score': 0.9991914629936218}, {'label': 'POSITIVE', 'score': 0.9983250498771667}, {'label': 'POSITIVE', 'score': 0.9997956156730652}, {'label': 'NEGATIVE', 'score': 0.9995220899581909}, {'label': 'NEGATIVE', 'score': 0.9992743134498596}, {'label': 'NEGATIVE', 'score': 0.9990591406822205}, {'label': 'NEGATIVE', 'score': 0.672651469707489}, {'label': 'NEGATIVE', 'score': 0.9981521964073181}, {'label': 'NEGATIVE', 'score': 0.9959521293640137}, {'label': 'NEGATIVE', 'score': 0.9990072846412659}, {'label': 'POSITIVE', 'score': 0.9994705319404602}]
Text: Oh, fuck me. I've just returned from the Supermarket Of Doom to find that I have nothing to drink here., Sentiment: NEGATIVE, score: 0.9991938471794128
Text: Moving office tomorrow after 3 years at this one, its a sad day, Sentiment: NEGATIVE, score: 0.8212727308273315
Text: #thingsmummysaid...rollercoasters are only multi-storey car parks without walls., Sentiment: NEGATIVE, score: 0.9679269790649414
Text: Killed a pigeon today  Thought it was going to move out the way of the car.. next thing I know, BANG feathers in the rear view mirror RIP, Sentiment: NEGATIVE, score: 0.9984161853790283
Text: @gk2007 Yu th? trang nï¿½y xem  http://bit.ly/kMxHk (recommended by ), Sentiment: NEGATIVE, score: 0.851008951663971
Text: just starting my day...a long Friday, Sentiment: NEGATIVE, score: 0.7902448773384094
Text: sooooo, i just dropped my phone.  don't text or tweet me, i'm currently banging my face against a spike covered poison ivy infested wall., Sentiment: NEGATIVE, score: 0.9991554021835327
Text: Goin to bed. Goodnight everyone., Sentiment: NEGATIVE, score: 0.897927463054657
Text: Thinks FML should changed to LML (love my life), Sentiment: NEGATIVE, score: 0.9929895401000977
Text: may the 4th be with you! HAPPY STAR WARS DAY!, Sentiment: POSITIVE, score: 0.999870777130127
Text: longest flight EVER. not particularly unpleasant or uncomfortable, just really really long, Sentiment: NEGATIVE, score: 0.9920926690101624
Text: @JonathanRKnight without ur tweets i feel lost hit me with something, Sentiment: POSITIVE, score: 0.9535270929336548
Text: @LLCee I had to find out via twitter, Sentiment: NEGATIVE, score: 0.9953281879425049
Text: @cloudconnected Actually I think the NA release date was confirmed for September so it's a bit more of a wait., Sentiment: NEGATIVE, score: 0.9991187453269958
Text: @bkGirlFriday thanks! You're the first one to wish me a happy mother's day, Sentiment: POSITIVE, score: 0.9995375871658325
Text: @JorinCowley I see. I guess there must be lots of Hawks fans in Texas on twitter., Sentiment: POSITIVE, score: 0.9633103013038635
Text: @jroberson4 Good luck at the services tomorrow!! I wish I could see you guys on your vacation!, Sentiment: POSITIVE, score: 0.9995717406272888
Text: @machfairy dont be gloomy...go out and get urself ice-cream.or gin,whichever, Sentiment: NEGATIVE, score: 0.9942452311515808
Text: Through to quaterfinals of charity football tournament. My penalty save sent us through, Sentiment: NEGATIVE, score: 0.6153771877288818
Text: I'm wishing I was outside instead of trapped in my office., Sentiment: NEGATIVE, score: 0.9924334287643433
Text: @DonnieWahlberg BTW I STILL can't believe how Awesome the NEWJABBAKIDZ performance was...U in the masks..I screamed at my pc, Sentiment: POSITIVE, score: 0.9462117552757263
Text: IT'S MOTHER'S DAY, Sentiment: POSITIVE, score: 0.9957706332206726
Text: closeness or distance? closeness... but now everything seems so distant..., Sentiment: NEGATIVE, score: 0.9997096657752991
Text: @ lovelytrinkets I like the way you worded that about Rocky Road, Sentiment: POSITIVE, score: 0.9996337890625
Text: @ChimeraX *Hand up* Me, I'm going  #localgovcamp, Sentiment: POSITIVE, score: 0.9785970449447632
Text: @Gabrielle_Union don't let anyone run you away from anything, Sentiment: POSITIVE, score: 0.9965008497238159
Text: @jbgreece yeh  A little.. How are you ?, Sentiment: NEGATIVE, score: 0.97406005859375
Text: is pretty dang tired. but chambers class is for napping., Sentiment: NEGATIVE, score: 0.9680083990097046
Text: Happy Star Wars Day ...  &quot;May the 4th be with you&quot;.... read http://tinyurl.com/axsujx for more ;), Sentiment: POSITIVE, score: 0.9996532201766968
Text: Moment over.... fly now in car, Sentiment: POSITIVE, score: 0.8575786352157593
Text: @jaceypants well piss on that. I can't get into their site @ work nor does it come thru on my phone., Sentiment: NEGATIVE, score: 0.9990264177322388
Text: i cant spell, Sentiment: POSITIVE, score: 0.9726644158363342
Text: Oh dang! 'Drag Me To Hell' came out today, didn't it? Man, I wish I remembered; I would have gone and seen it., Sentiment: NEGATIVE, score: 0.9972148537635803
Text: Someone fly me to Reno, Sentiment: POSITIVE, score: 0.9844653010368347
Text: im officially done with school til fall., Sentiment: POSITIVE, score: 0.8398813605308533
Text: The computers and the Ethernet at school are so slow!, Sentiment: NEGATIVE, score: 0.9997662901878357
Text: is missing out on the sunshine and trying to stay awake after having just 2 hours sleep, Sentiment: NEGATIVE, score: 0.9996846914291382
Text: I've got sunburn on my arm  In better news, my new Guitar Hero: Metallica game came and I beyond happy about that., Sentiment: POSITIVE, score: 0.9348998665809631
Text: Waitin for the man to get home so he can take me out !!!! been waitin 4 hours, Sentiment: NEGATIVE, score: 0.9954380393028259
Text: @simonwilder I want to play, Sentiment: POSITIVE, score: 0.998193085193634
Text: @iheartrendering awww. its cool. i ate too much ice cream, Sentiment: NEGATIVE, score: 0.9846375584602356
Text: @justgrimes - yep saw that paper immediately after completing the test essay saying i didn't know of research on it, Sentiment: NEGATIVE, score: 0.9986984729766846
Text: is at home, Sentiment: POSITIVE, score: 0.9997170567512512
Text: I miss my mom..  &quot;May angels lead you in&quot;, Sentiment: NEGATIVE, score: 0.9974105954170227
Text: @GabrielSaporta @SUAREASY @NovarroNate you guys were absolutely amazing tonight, as always. thanks for always bringing the dance party., Sentiment: POSITIVE, score: 0.9997467398643494
Text: Good morning everyone  It's a nice day #iloveitwhen the sun is shining. And now I'm going to write some stuff, Sentiment: POSITIVE, score: 0.9998138546943665
Text: RIP Omar Edwards - Killed by friendly fire in NYC   http://bit.ly/jrM6v, Sentiment: NEGATIVE, score: 0.9981138706207275
Text: My dad just told me that he wants to put me up for sale on craigslist, Sentiment: NEGATIVE, score: 0.9971684813499451
Text: watching run fat boy run...haha its soo funny., Sentiment: NEGATIVE, score: 0.8268777132034302
Text: @megan_ward i am, Sentiment: POSITIVE, score: 0.8898416757583618
Text: is excited about Taylor Swift on wednesday!!!, Sentiment: POSITIVE, score: 0.9997978806495667
Text: http://twitpic.com/665w2 - see miles away, Sentiment: NEGATIVE, score: 0.9987711310386658
Text: healthy food is NOT helping my hangover, Sentiment: NEGATIVE, score: 0.9962769150733948
Text: Dilemma, what to wear: Now: SanFran Foggy and 58 , then Sacramento in cple hrs sun and 86 degr ., Sentiment: NEGATIVE, score: 0.9958756566047668
Text: @wendy_fred6 Awww... :/ I guess that's both good and bad, moving is not an option I guess?  Mhm, so are you, hehe ;) (we have same time?), Sentiment: NEGATIVE, score: 0.9978819489479065
Text: Taking Horse Pills, hoping I can get some sleep tonight, Sentiment: NEGATIVE, score: 0.999014139175415
Text: school then the used concert tonight!!, Sentiment: NEGATIVE, score: 0.9339135885238647
Text: Metro from trader joe to 71st closed so many firemen and cops wth happened?! Walking home bus can't go further  ohh, Sentiment: NEGATIVE, score: 0.9943027496337891
Text: I'm really tired today - I must have slept very badly... I'm glad it's an &quot;off&quot; Friday, but I've still got a ton of stuff to get done  #fb, Sentiment: NEGATIVE, score: 0.9985378980636597
Text: Cannot tweet. Eyes still dilated from morning eye exam. Am on verge of bifocals, and so is @adravan, Sentiment: NEGATIVE, score: 0.975536048412323
Text: winding down, love having a low key day., Sentiment: POSITIVE, score: 0.9933941960334778
Text: I know I shouldn't be saying this but fuck it..I'm horny as hell  http://twurl.nl/8q6cjc, Sentiment: NEGATIVE, score: 0.9984363913536072
Text: Now watching ZDF Fernsehgarten. Its so great that Andrea is back, at last, Sentiment: POSITIVE, score: 0.9997119307518005
Text: at school right now, Sentiment: POSITIVE, score: 0.996401309967041
Text: Mom says I have to get a new phone IMMEDIATELY....off to T-Mobile.  she paying...., Sentiment: NEGATIVE, score: 0.998042106628418
Text: @Charified sadly, I don't   hehe, Sentiment: NEGATIVE, score: 0.9936827421188354
Text: 48 days till brighton, Sentiment: POSITIVE, score: 0.9427001476287842
Text: starting to wonder if I'm going to get this job... came across as though i'd find work elsewhere if needs must... not intentionally, Sentiment: NEGATIVE, score: 0.9996917247772217
Text: @RedMummy And it's such glorious weather too - poor you, Sentiment: POSITIVE, score: 0.9980125427246094
Text: just got back from the pool, need to ice the knee, Sentiment: NEGATIVE, score: 0.9982934594154358
Text: so...i really want to be home right now., Sentiment: POSITIVE, score: 0.9991291165351868
Text: URL in previous post (to timer job) should be http://bit.ly/a4Fdb. I'd removed space which messed up URL.  ^ES, Sentiment: NEGATIVE, score: 0.999555766582489
Text: lees net op Twitter dat het #Happy Star Wars day is... &quot;May the 4th be with you&quot;... Sjeez wat slecht, Sentiment: NEGATIVE, score: 0.9117883443832397
Text: @npyskater Thank you!, Sentiment: POSITIVE, score: 0.9996329545974731
Text: What if Twitter was really called &quot;Twatter&quot;? I'm posting a Twat!, Sentiment: NEGATIVE, score: 0.9992640614509583
Text: @DavidBurke1 morning David have a safe journey and enjoy your time in the states  xxx, Sentiment: POSITIVE, score: 0.9994450211524963
Text: @isuhin O dear. so you're going to be fucking that kiddoe and I'm going to sit there doing..  nothing! awesome  WHERES THE SHOPPING?!&lt;3, Sentiment: NEGATIVE, score: 0.9962683320045471
Text: @suewaters Sorry - I have failed to grasp your meaning, Sentiment: NEGATIVE, score: 0.999613344669342
Text: trying to figure this twitter thing out!   I'm quite excited about it, Sentiment: POSITIVE, score: 0.9995705485343933
Text: @CKHerm Glad you got to walk.  Finish the damn thesis., Sentiment: POSITIVE, score: 0.6506785750389099
Text: @gabriellenadine carnivalsofparis i think i still have yours on my bl but i'm not sure... i hardly talk to anyone anymore, Sentiment: NEGATIVE, score: 0.9989368319511414
Text: I missed you yesterday, Lacey.    We get to go to granulation tonight, though., Sentiment: NEGATIVE, score: 0.7903785109519958
Text: @imkeshav I love flock on ubuntu, Sentiment: POSITIVE, score: 0.9975118637084961
Text: @SeanyeWest the mind plugs work, but the patent got turned down b/c someone already thought of earplugs and benedryl, Sentiment: NEGATIVE, score: 0.9991369843482971
Text: @moanasaves glad u liked post  looking at the back end now...the sub 2 your blog is automatic. send url and i'll verify., Sentiment: NEGATIVE, score: 0.9908849596977234
Text: TIRED! goodnight twitter  its mother's day  happy mother's day  lov my moomy &lt;3 yayy! God Bless., Sentiment: NEGATIVE, score: 0.8972830772399902
Text: @taylorswift13 i wish you could come to Swindon...2 hours away...its my dream to meet you  xoxo, Sentiment: POSITIVE, score: 0.9835255146026611
Text: @laracasey: LOVE you walking us through this event. So fun. Blue water. Surplus of donuts... what a great night!, Sentiment: POSITIVE, score: 0.9998327493667603
Text: @avenueofthearts My Pleasure, Sentiment: POSITIVE, score: 0.9996351003646851
Text: Oh yeah, Radio1 is SO playing Earth, Wind and Fire, Sentiment: POSITIVE, score: 0.972537100315094
Text: it's @andreamichellef's birthday today, wish her a good one assholes! Sleepytime, Sentiment: POSITIVE, score: 0.9573084712028503
Text: @miniatus Well thank you darling...it was a pleasure shopping with you...you will see the first pics!, Sentiment: POSITIVE, score: 0.9997953772544861
Text: @GabrieleDurning  sadly we were too late for TraceyCakes, but we got them elsewhere- not the same but still yummy. Tea w wee on wknd!, Sentiment: POSITIVE, score: 0.9649789333343506
Text: ugh i need a job but no one is hiring, Sentiment: NEGATIVE, score: 0.9988241791725159
Text: Watering the plants at home. Drinking a delicious smoothie from morgans because my jamba exploded., Sentiment: POSITIVE, score: 0.9991419315338135
Text: @trueblooddallas Dallas, I have a few Questions for you but, can't direct to you cause your not following me, Sentiment: NEGATIVE, score: 0.9989161491394043
Text: @SarahWV  worse case scenario i'll take tomorrow am off. are you still up or you just woke up?, Sentiment: NEGATIVE, score: 0.9993619322776794
Text: Must head back to the office, Sentiment: NEGATIVE, score: 0.9822791814804077
Text: @LeslieLang It will be an Adventure!  Have FUN with your 5 yr old and 8 mo old! (You're brave.)    Hope you have a great time!, Sentiment: POSITIVE, score: 0.9998759031295776
Text: @cccaaasss what's wrong?, Sentiment: NEGATIVE, score: 0.9991922974586487
Text: thinks she needs more followers. its still so warm  going cinema later t c night at the museum2, Sentiment: NEGATIVE, score: 0.9807064533233643
Text: @donperignon me too baby... Miss you, Sentiment: NEGATIVE, score: 0.7379233837127686
Text: @limeice arre main toh bakwaas kar raha tha. Sunday morning bakwaas, Sentiment: NEGATIVE, score: 0.9959179759025574
Text: @theellenshow get @kalebnation the twilightguy on your show, Sentiment: NEGATIVE, score: 0.9805828928947449
Text: Awww my daddy! Got in a car accident! Pray for him! He's shook'n up a lil!, Sentiment: NEGATIVE, score: 0.9890885353088379
Text: thank god i havent quit my day job  ps. turning 27 tomorrow. i just round up to 30 now., Sentiment: POSITIVE, score: 0.980602502822876
Text: I'm a lil sad looks like nomore brooklyn 4 a while WTF @lailashah, Sentiment: NEGATIVE, score: 0.9967902302742004
Text: wanted to go to white sands today. forecast says there will be thunderstorms..., Sentiment: POSITIVE, score: 0.903872549533844
Text: @caniszczyk Agreed! Though Eclipse apps hinder collecting the heap dump by catching OOME. Had to muck about in JConsole, Sentiment: NEGATIVE, score: 0.9951327443122864
Text: home alone and no one left me any gummy bears, Sentiment: NEGATIVE, score: 0.995582640171051
Text: @LeMonjat Hehe, funny (the midget thing) ! Cheer up Alex, and wave from below  Is it that you are in Germany right now? .. or in Spain? ;D, Sentiment: POSITIVE, score: 0.9977433681488037
Text: IM FEELIN RITE.. THE MOOD FLOR TWITTER AFTER DARK..., Sentiment: NEGATIVE, score: 0.8919761776924133
Text: @katelynizzle haha okay you were talking about middle college grad I think and I got worried, Sentiment: NEGATIVE, score: 0.9932039976119995
Text: @JessicaTGolden yeah I LOVE CALI so much, Sentiment: POSITIVE, score: 0.9988875985145569
Text: i'm celebrating my mother!!  and also celebrating my legacy as a woman of God., Sentiment: POSITIVE, score: 0.9998418092727661
Text: @Brawny2004 true true, I'm writing atm, trying 2 b coherent about the last 4 yrs&amp;string narratives through it but my meats rotting nicely, Sentiment: NEGATIVE, score: 0.9492014050483704
Text: Getting ready for week  Its too nice today to be stuck inside working!, Sentiment: NEGATIVE, score: 0.9946106672286987
Text: Dan and alli are here. They suprised me, Sentiment: NEGATIVE, score: 0.8950132131576538
Text: @mrs_mcsupergirl ok, finished set u free, and i am sooooo mad rite now...it cant end like that, i dont want him to be the bad guy, Sentiment: NEGATIVE, score: 0.9602644443511963
Text: @nomadiquemc I want to be at @urbangrind, Sentiment: POSITIVE, score: 0.8336657285690308
Text: @carinacani DAMN @-) That's a lot of messages from him @-) HOW SWEET. :&quot;&gt; And yeah, sayang, Sentiment: POSITIVE, score: 0.9995779395103455
Text: On the way back to dublin Omg didnt hit the bed until 530  so i am so sleepy   but once again on the road back to good ole  dublin :-p ..., Sentiment: NEGATIVE, score: 0.9358395934104919
Text: @mrskutcher you're so classy, demi. Love it, don't stop doing your thing., Sentiment: POSITIVE, score: 0.9998352527618408
Text: got an RE exam on Tuesday. Wish me luck / pray for me? Thank you.  xxx, Sentiment: POSITIVE, score: 0.9176763296127319
Text: FML. I hate CSS SO BAD. I can't find an lj layout that has everything I want., Sentiment: NEGATIVE, score: 0.999453604221344
Text: @lesley007 morning sweetie, you cool?  xxx, Sentiment: POSITIVE, score: 0.9857500791549683
Text: @pamjob yay  i'll do you a heart mk shout in a sec pam, Sentiment: POSITIVE, score: 0.9903451800346375
Text: @lejjewellery oh nice going!, Sentiment: POSITIVE, score: 0.9996941089630127
Text: @IsaacMascote  i'm sorry people are so rude to you, isaac, they should get some manners and know better than to be so lewd!, Sentiment: NEGATIVE, score: 0.9973114728927612
Text: ...getting our site transferred over to a new server ... this is going to be quite a job, Sentiment: POSITIVE, score: 0.994162380695343
Text: @letskilldave - Yea, I really need to learn to reload my own, Sentiment: NEGATIVE, score: 0.9983218312263489
Text: My job sucks!!!, Sentiment: NEGATIVE, score: 0.9990423321723938
Text: I WILL CRY!!!!!!!!    I can't believe that I lost the chat!!!, Sentiment: NEGATIVE, score: 0.9979262351989746
Text: @hollypop04 mmm  Where is it Holly?, Sentiment: NEGATIVE, score: 0.9953429698944092
Text: Bah DHCP server, why must you keep falling on your face, Sentiment: NEGATIVE, score: 0.9991901516914368
Text: @myucan91 wahahahaha!! i wanna naaaaa!!! well...hapit na i guess. hahahaha ) yes, now we all know!!! hahahaha lol ) NARN! haha joke, Sentiment: NEGATIVE, score: 0.9976270794868469
Text: ready to go home, Sentiment: POSITIVE, score: 0.9993953704833984
Text: WFD: Lasagna. Still 45 minutes to go, so hungry now., Sentiment: NEGATIVE, score: 0.9906971454620361
Text: @mikeshelby Now that's a very nice way to fall asleep., Sentiment: NEGATIVE, score: 0.9825665354728699
Text: @ntpro Hmm.  My VPN works fine.    (Oh.. wait.. I don't need VPN anymore.) http://tinyurl.com/cao6tu, Sentiment: POSITIVE, score: 0.9831072092056274
Text: Can't wait 2 hand in work tomorrow then im practically finished for the year  yay!!!!!!!, Sentiment: POSITIVE, score: 0.9589449763298035
Text: Cooking dinner!! Its already late!! am making Cabbage Molagootal for dinner!!, Sentiment: NEGATIVE, score: 0.9782748222351074
Text: You spelled my name wrong, but message received   http://tinyurl.com/krw9p3, Sentiment: NEGATIVE, score: 0.9980385899543762
Text: @ccr_harris  There were way more than two! Ten hours of real-ale takes it out of you, Sentiment: NEGATIVE, score: 0.9677836298942566
Text: @iamyoushouldtoo Oh, I'm jealous (how surprising), Sentiment: POSITIVE, score: 0.9936049580574036
Text: @beccaRAR I like to support my friends  It's sad that I'm your only friend though, Sentiment: NEGATIVE, score: 0.6863706707954407
Text: @KarlaaM_ A blouse! ahahaha I gave her money and she went to get it!  Where's your mom??, Sentiment: NEGATIVE, score: 0.985194981098175
Text: @pmcclory hmm. Tough choice. You got some matches?, Sentiment: NEGATIVE, score: 0.9861451387405396
Text: did not get to go see UP!! Oh well ended up going to dinner with Blase and Bridget!, Sentiment: NEGATIVE, score: 0.9929673671722412
Text: Off to work, Sentiment: NEGATIVE, score: 0.8717076778411865
Text: @Broooooke_ listen to FTSK  they stop my bordum  haha how was your day? finished crying about Harold? ha xx, Sentiment: NEGATIVE, score: 0.9986456036567688
Text: id be happy thats its friday if i didnt have to work tomorrow  blah, Sentiment: POSITIVE, score: 0.9912693500518799
Text: @urbansmiler Is it possible to a have phobia of phobias? Afraid to look at list., Sentiment: NEGATIVE, score: 0.9971727132797241
Text: I don't ship out until October, Sentiment: NEGATIVE, score: 0.8640374541282654
Text: ahhh ... i don't care, i love this movie in all it's cheesy-ness, Sentiment: POSITIVE, score: 0.9965356588363647
Text: May the 4th be with you! Happy Star Wars Day twirps!  ROFL, Sentiment: POSITIVE, score: 0.9996576309204102
Text: Aw, not going to Toronto anymore., Sentiment: NEGATIVE, score: 0.9978697299957275
Text: @Mitchelmusso: I sent you an other call me back message  x, Sentiment: NEGATIVE, score: 0.9906909465789795
Text: Wow.....I've been stood up  Might as well go to work., Sentiment: POSITIVE, score: 0.994864284992218
Text: Last day of high school!, Sentiment: POSITIVE, score: 0.9955570101737976
Text: just finished designing her multiply site, Sentiment: NEGATIVE, score: 0.9880920648574829
Text: Good morning friends. Happy May Bank Holiday., Sentiment: POSITIVE, score: 0.9998648166656494
Text: @worldofhiglet I don't think it makes you seem shallow. Some actually do respond and converse with followers.  If they don't, no biggie., Sentiment: NEGATIVE, score: 0.6356391310691833
Text: taking requests for nkkairplay, Sentiment: NEGATIVE, score: 0.9892799258232117
Text: @TomFelton Safe flight home to you and Jade   XX, Sentiment: POSITIVE, score: 0.996052086353302
Text: Broke the laptop again..., Sentiment: NEGATIVE, score: 0.9968383312225342
Text: Looking forward to having Dinner with Family and Friendsss! Happy Mothers Day to all the Moms out there!, Sentiment: POSITIVE, score: 0.999839186668396
Text: @DonnaFirsty she fell into deep crack in the glacier  so terrible, Sentiment: NEGATIVE, score: 0.9992460012435913
Text: Omg its so gross out. no  relays tonight!, Sentiment: NEGATIVE, score: 0.9992926120758057
Text: In the middle of breakfast the school called.Yep...back to get Shey...AGAIN.3rd migraine this week or maybe one that never went away, Sentiment: NEGATIVE, score: 0.9726062417030334
Text: @GoonersNato - Aww, yeah, dear Nicky  ., Sentiment: POSITIVE, score: 0.9289051294326782
Text: @Nicsey Snap! I know that feeling well, Sentiment: POSITIVE, score: 0.9983618855476379
Text: @sandy195850 we have two small dogs, good to hear that Center Parcs wd take them. Cruising to New York on the Queen Mary 2, real treat, Sentiment: POSITIVE, score: 0.9993817806243896
Text: Good morning Monday...  I feel as though you came to visit too early... But I am happy to see you none the less., Sentiment: POSITIVE, score: 0.9995130300521851
Text: @chriscornell Thank you!  The best to you and yours tomorrow... I hope you all have a lovely day together!!, Sentiment: POSITIVE, score: 0.9998675584793091
Text: @mcrfash1 cool  what did you get?, Sentiment: POSITIVE, score: 0.8328090906143188
Text: Just rang the irish one. Drunk. Must confiscate phone.  hate him lots., Sentiment: NEGATIVE, score: 0.9996616840362549
Text: Just found out one of my coworkers that I actually like is leaving., Sentiment: NEGATIVE, score: 0.9979817867279053
Text: So my alarm got changed somehow and I ended up waking up at 1:00  I feel like half my day is gone, Sentiment: NEGATIVE, score: 0.9996119141578674
Text: @CRISCOKIDD this picture made me cry  lol http://www.twitpic.com/671w1, Sentiment: NEGATIVE, score: 0.9946340322494507
Text: Doctors appt, Sentiment: NEGATIVE, score: 0.5661691427230835
Text: Hate being skint  Anybody want to give me another job? haha!, Sentiment: NEGATIVE, score: 0.9498242139816284
Text: hiccups, Sentiment: NEGATIVE, score: 0.952165424823761
Text: @Keels_90 haha agreed  LOL, Sentiment: NEGATIVE, score: 0.9660447239875793
Text: finally home for once after a dope ass week, Sentiment: POSITIVE, score: 0.831809401512146
Text: @lisisilveira   I sent my donation to #Eric and wanted to put the banner on, but my avatar disappeared when I tried ~, Sentiment: NEGATIVE, score: 0.9989569187164307
Text: I really really really dont want to go to work! 4th shift of the week just to start over on sunday, Sentiment: NEGATIVE, score: 0.9828109741210938
Text: @boredzo I assume you mean 2nd item with 3 nested lines, but no invert call tree button to be found   ï¿½  ï¿½   ï¿½, Sentiment: NEGATIVE, score: 0.998397171497345
Text: @stevencohmer -thanks i hope i do 2  iv been playing dmc4 like 5 times kinda sick of it hehe, Sentiment: NEGATIVE, score: 0.9981616139411926
Text: Venus Williams is having a horrible day at the office, Sentiment: NEGATIVE, score: 0.9991914629936218
Text: @citycynic Sounds good to me! No more cleaning cynics orders. Haha. Good night. Talk tomorrow., Sentiment: POSITIVE, score: 0.9983250498771667
Text: Had an awesome pedicure today!!, Sentiment: POSITIVE, score: 0.9997956156730652
Text: I took my yearbook photo earlier at school, and I don't think that it will turn out great, Sentiment: NEGATIVE, score: 0.9995220899581909
Text: @xsparkage imagine if you really were lost and lost all contact with DT!  that'd be the saddest day of my life haha, Sentiment: NEGATIVE, score: 0.9992743134498596
Text: @tommcfly Tom are the MITO tour dvd's still being made cos no where seem to be stealing them and i really want a copy, Sentiment: NEGATIVE, score: 0.9990591406822205
Text: @Jae878  thanks  I'm holding my mini laser light thingy lol. How are u?, Sentiment: NEGATIVE, score: 0.672651469707489
Text: @Patti0713 It's not on my cable carrier. Only Space channel, Sentiment: NEGATIVE, score: 0.9981521964073181
Text: is cheering on the Arsenal Ladies!!    and missing my M...x, Sentiment: NEGATIVE, score: 0.9959521293640137
Text: The only thing about preparing 5 sermons in one week is just when you think you are done...you have to prepare the powerpoint slides, Sentiment: NEGATIVE, score: 0.9990072846412659
Text: @Heart_song Glad you had a wondrous Beltaine! Mine was quiet, yet in sync with the season.  Miss you all..., Sentiment: POSITIVE, score: 0.9994705319404602
```
---
