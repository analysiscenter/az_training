- общее устройство и основные принципы (+), git init (+), git add (+), git rm, git commit, git status, git diff, как следует писать commit-сообщения
- git config (+), git remote, git fetch, git pull, git push, gitattributes, gitignore



# Общее устройство и основные принципы

## немного мотивации и происхождение названия

- хочется хранить старые копии файлов, чтобы можно было откатиться, в случае чего
--- можно копировать старые файлы - легко запутаться, забыть скопировать, неэффективно по памяти
--- можно воспользоваться системой контроля версий -- например, git - Distributed Version Control Systems -- на каждой машине хранится копия _всего_ репозитория.

The name "git" was given by Linus Torvalds when he wrote the very
first version. He described the tool as "the stupid content tracker"
and the name as (depending on your way):

 - random three-letter combination that is pronounceable, and not actually used by any common UNIX command. The fact that it is a mispronunciation of "get" may or may not be relevant.
 - stupid. contemptible and despicable. simple. Take your pick from the dictionary of slang.
 - "global information tracker": you're in a good mood, and it actually works for you. Angels sing, and a light suddenly fills the room.
 - "goddamn idiotic truckload of shit": when it breaks


## основное устройство

- В отличие от многих систем контроля версий, git не хранит изменения во времени для каждого файла. На самом деле хранятся снапшоты -- копии всей файловой системы. Если файл в следующем снапшоте не менялся, то хранится ссылка на этот файл из предыдущего снапшота. (картинка https://git-scm.com/book/en/v2/images/snapshots.png)
- Почти каджая операция происходит на локальной машине, так как вся история хранится локально
- Для каждого файла git хранит хэш -- последовательность из 40 чисел в 16-ричной системе -- по ним и определяется, менялся ли файл
- файлы в гите могут находится в трех разных состояниях (картинка https://git-scm.com/book/en/v2/images/areas.png):
-- Commited - файлы сохранены в локальной БД
-- Modified - файлы изменены, но изменения не внесены в локальную БД
-- Staged - файлы помечены, будут включены в следующий коммит
- Соответственно, в гите есть три разных "пространства": 
-- repository -- самое важное. Хранятся метаданные и БД для проекта. Именно это копируется, когда клонируем репозиторий (об этом далее)
-- working directory -- локальная копия одной версии проекта, с которой происходит работа
-- staging area (или index) -- файл, содержащий информацию о том, что пойдет в следующий коммит
- Стандартно, работа в гите выглядлит так:
-- Меняем файлы в working directory
-- Указываем в staging area, какие именно изменения будут учтены при следующем коммите
-- Делаем коммит, то есть записываем снапшот из staging area в БД (commited).

# Базовые команды

- есть вопрос по команде? набираем

-- $ git help <имя команды>

- Прежде чем начать работать с гитом, нужно настроить окружение

-- для этого есть git config -- утилита для управления переменными окружения (имя пользователя и тп). Настроим имя пользователя и почту -- это будет включено в каждый совершенный коммит (global означает, что переменные будут использоваться всегда и везде, если хочется сменить их для специфичного проекта, просто опусти его):

--- $ git config --global user.name "John Doe"
--- $ git config --global user.email johndoe@example.com

-- проверить настройки:

--- $ git config --list

- Как создать репозиторий:

-- Можно зайти в папку и сказать, что теперь это репозиторий

--- $ cd folder
--- $ git init

--- Теперь в folder появилась папка .git -- сам репозитоий
--- Начнем отслеживать изменение файлов, для этого добавим их (то есть сделаем их staged):

--- $ git add some_file
--- $ git add \*.py --  можно добавлять по маске

--- Теперь добавим их в БД (сделаем их commited)

--- $ git commit -m 'initialized project repository'

--- (!)НАПИСАТЬ, КАК ПРАВИЛЬНО ПИСАТЬ СООБЩЕНИЯ

-- А можно _склонировать_ репозиторий откуда-нибудь (не забываем, что копируется _весь_ репозиторий, а не только текущая версия проекта)

--- $ git clone https://github.com/analysiscenter/dataset.git folder

--- Последняя команда создаст папку folder, инициализирует внутри репозиторий .git (и загрузит все его данные) и загрузит текущую рабочую копию проекта (если убрать folder, то именем папки будет имя проекта -- dataset).
