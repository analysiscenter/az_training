
<!-- И чем он большое, тем лучше сеть учитывает корреляции пикселей. -->
 The more - the merrier, meaning that | network takes into consideration this pixel correlation.!!

<!-- Увеличить rf можно используя большие свертки, но это требует больших вычислительных мощностей. -->
Increasing RF can be obviously reached by enlarging convolutions |, but this consumes a lot of computational capabilities.

<!-- Авторы VGG доказали, что можно получить бОльший rf применяя много последовательных сверток малого размера. Таким образом две подряд идущие свертки 3х3 заменяют одну 5х5, и при этом они требуют примерно на 28% меньше памяти и, соответственно, требуют меньше вычислений. -->
VGG creators proved community wrong, |b wey achieving lArger RF with applying multiple sequential convolutions of smaller size. This way 2 sequential 3x3 convolutions substitute 1 5x5 and by doing so, they as well save approximately 28% less of memory and, respectively - less computations.

<!-- В своей статье авторы приводят 6 различных вариаций VGG от 11 до 19 слоев, включая 4 полносвязных слоя в конце. В среднем во всех её реализациях около 140 м параметров. Из них только 24% весов содержатся в сверточных слоях, а остальные параметры находятся в полносвязных слоях. -->
In their outcome, authors showcase 6 different VGG variations of 11 to 19 layers, including 4 fully-connected layers in the end. On average, each one of its realisations contains around 140 mil parameters. Among all of them only 24% of weights are contained in the convolutional layers, while the rest are maintained in fully-convolutional.

<!-- Подводя итоги давайте посмотрим на плюсы и минусы этой архитектуры: Сначала поговорим о достоинствах: -->
We’ve went thought major features and to make it even we should summarise our review result. Let’s have a look at bonuses and limitations. Sure thing - bonuses first:

<!-- * Во-первых, использование маленьких сверток позволяют увеличить rf, не сильно увеличивая кол-во параметров. -->
Feature 1 - usage of smaller convolutions allows to increase RF, meanwhile keep number of parameters even

<!-- * Во-вторых, использование двух сверточных слоев, вместо одного позволяет делать два нелинейных преобразования, таким образом увеличивая пространство поиска. -->
Feature 2 - enjoyment of 2 convolutional layers, instead of 1 - allows execution of 2, instead of 1, non-linear transformations, thus enhancing the ________(пространство поиска)

<!-- *И в третьих. Архитектура очень проста в реализации. -->
Feature 3 - most important, the architecture is UBER-easy to replicate.

<!-- Теперь о недостатках: -->
Now, check-out the limits

<!-- * Если сделать сеть глубже, то появится проблема затухания градиента.? О которой мы поговорим в следующем видео? -->
If you decide to increase the depth of the net - you will, most definitely meet the vanishing gradients проблема затухания градиента, which is the subject of our next video

* Еще одним существенным недостатком является большое количество параметров в полносвязных слоях. Отсюда вытекает новая проблема - сеть медленно обучается. Даже сами авторы обучали эту сеть около двух недель на 4-х nvidia titan.

Благодаря простоте реализации сейчас существует множество уже обученных вариантов VGG.

Таким образом она остается актуальной до сих пор и используется в новых архитектурах для начальной обработки изображений, например в задачах сегментации (SSD или FCN-8).