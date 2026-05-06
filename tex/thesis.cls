\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{thesis}[]

\def\thesislanguage#1{\gdef\@thesislanguage{#1}}

\DeclareOption{Ingles}{
  \thesislanguage{0}
  \typeout{thesis: english.}
}

\newif\iffinalversion
\DeclareOption{Final}{
  \finalversiontrue
  \typeout{thesis: versao final.}
}

\ProcessOptions\relax

% Base class
\iffinalversion
  \LoadClass[12pt,a4paper]{article}
  \RequirePackage{silence}
  \WarningsOff[biblatex]
\fi

% Required packages
\RequirePackage{xstring}
\RequirePackage[english]{babel}
\RequirePackage{fancyhdr}
\RequirePackage[ttscale=.875]{libertine}
\RequirePackage{libertinust1math}
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{graphicx}
\graphicspath{{./logotipos/}{./figuras/}}
\PassOptionsToPackage{table}{xcolor}
\RequirePackage[
        pdfdisplaydoctitle = true, 
        pdfstartview={XYZ null null .75}
    ]{hyperref}
\RequirePackage{nomencl} \makenomenclature
\RequirePackage{xspace}
\RequirePackage{setspace}
\RequirePackage{geometry}
\geometry{a4paper,top=30mm,bottom=20mm,left=30mm,right=20mm}
\RequirePackage{csquotes}
\RequirePackage[style=abnt,uniquename=init,giveninits,uniquelist=true,maxbibnames=4,repeatfields=true,justify]{biblatex}
\renewcommand*{\UpperOrSC}[1]{#1}
\renewcommand*{\UpperOrSCCite}[1]{#1}
\RequirePackage{booktabs}
\renewcommand{\arraystretch}{1.2}

% Language specific settings
\ifcase\@thesislanguage\relax
  %English
  \RequirePackage[english,nameinlink]{cleveref}
\fi
  \Crefformat{equation}{#2\Cref@equation@name~#1#3}
  \creflabelformat{equation}{#2\textup{#1}#3}

% New commands and variables
\newcommand{\@CIDADE}{Campinas}
\newcommand{\@UNIDADEENSINO}{Faculdade de Engenharia Elétrica e de Computação}
\newcommand{\@UNIDADEENSINOABREV}{FEEC}
\newcommand{\@UNICAMP}{Universidade Estadual de Campinas}

% Keywords command
\newcommand{\keywords}[1]{\gdef\@keywords{#1}}
\let\@keywords\@empty % Initialize as empty

\let\@autor\relax
\def\autor#1{\gdef\@autor{#1}}
\author{\@autor}

\let\@titulo\relax
\def\titulo#1{\gdef\@titulo{#1}}
\title{\@titulo}

% Document type definitions
\gdef\qualificacaoMestrado{
  \gdef\@textoaviso{\clearpage}
  \gdef\@degnamept{Mestre em Engenharia Elétrica, na área de Engenharia de Computação}
  \gdef\@artigo{o}
  \gdef\@monopt{Monografia de qualificação}
  \gdef\@nivelEnsino{Pós-graduação}
}

% Advisor commands
\let\@orientador\relax
\def\orientador#1{\gdef\@orientador{#1}}

\gdef\@advisors{1}

% Date of defense
\def\datadadefesa#1#2#3{
  \gdef\@dia{#1}
  \gdef\@mes{#2}
  \gdef\@ano{#3}
}

% Area of concentration
\let\@areaConcentracao\relax
\def\areaConcentracao#1{\gdef\@areaConcentracao{#1}}

% Initial pages setup
\def\paginasiniciais{
  \newlength{\oldparindent}
  \setlength{\oldparindent}{\parindent}
  \setlength\parindent{0pt}
  \thispagestyle{empty}
  \pagestyle{empty}  
  \makeatletter
  \let\ps@mystyle\ps@plain
  \let\ps@plain\ps@empty
  
  % PDF properties
  \hypersetup{
    pdfcreator   = {Criado com o modelo para teses e dissertações da FT, elaborado pelo prof. André Leon S. Gradvohl, Dr.},
    pdftitle     = {\@titulo},
    pdfauthor    = {\@autor},
    pdfsubject   = {\@monopt\xspace apresentada à Faculdade de Tecnologia da Universidade Estadual de Campinas como parte dos requisitos para a obtenção do título de \@degnamept\xspace.}
  }
  
  \makeatother
  \openningpage
  \clearpage
  \titlepage
  \@textoaviso

  % Language specific titles
  \ifcase\@thesislanguage\relax
  %English
  \renewcommand{\nomname}{List of Acronyms}

  \setlength{\parindent}{\oldparindent}
  \resetlang
}

% Logos
\def\logos{
    {\parbox[c]{0cm}{\includegraphics[width=2.2cm]{logo-unicamp.pdf}}}
       {\parbox[c]{\textwidth}
       {
         \noindent
         \centering{
           \large \MakeUppercase{\@UNICAMP}\\
                   \@UNIDADEENSINO}
       }
    }
}

% Opening page
\def\openningpage{
  \logos
  \vskip 70mm
  \begin{center}
    \Large
    {\@autor}
    \vskip 25mm
    \ifcase\@thesislanguage
      {\bf\@title}
      \vskip 25mm
    \or
      {\bf\@title}
      \vskip 25mm
    \fi
    {\bf \@titulo}
    \vfill
    \large
    \@CIDADE\\\@ano
  \end{center}
}

% Title page
\def\titlepage{
  \large\bf
  \vfill
  \begin{center}
    \@autor
    \vfill
    \ifcase\@thesislanguage
      \@title
      \vskip 8mm
    \or
      \@title
      \vskip 8mm
    \fi
    \@titulo
  \end{center}
  \vfill
  \normalfont
  \normalsize
  \hfill\begin{minipage}[t]{0.54\textwidth}
    \supresshyphen
    \@monopt\xspace apresentada à Faculdade de Engenharia Elétrica e de Computação da Universidade Estadual de Campinas como parte dos requisitos para a obtenção do título de \@degnamept\xspace \ifx\@areaConcentracao\relax\else.
  \end{minipage}
  \vfill
  {\bf Orientador: \@orientador}
  \vfill
  \begin{center}
    \large
    \@CIDADE\\ \@ano
  \end{center}
}

% Reset language
\gdef\resetlang{
  \ifcase\@thesislanguage\relax
  \selectlanguage{english}
}

% Suppress hyphenation
\gdef\supresshyphen{
  \tolerance=1
  \emergencystretch=\maxdimen
  \hyphenpenalty=10000
  \hbadness=10000
}

% Pre-textual elements
\renewenvironment{abstract}{%
  \prefacesection{\centering\Large\bfseries Abstract}%
  \selectlanguage{english}%
  \setstretch{1.25}%
}{%
  \ifx\@keywords\@empty\else%
    \par\vspace{1.5em}%
    \noindent\begin{minipage}{\textwidth}
      \textbf{Keywords:} \@keywords%
    \end{minipage}%
  \fi%
  \resetlang%
  \newpage%
}

% Section formatting
\def\prefacesection#1{
  \section*{#1}
  \pagestyle{empty}
  \thispagestyle{empty}
}

% End of initial pages
\def\fimdaspaginasiniciais{
\newpage
\makeatletter
\let\ps@plain\ps@mystyle
\makeatother

% Page style definitions
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyhead[R]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\headheight}{14.49999pt}
}

\fancypagestyle{headings}{%
  \fancyhf{}
  \fancyhead[L]{\textsc{\nouppercase{\leftmark}}}
  \fancyhead[R]{\thepage}
  \renewcommand{\headrulewidth}{1pt}
  \renewcommand{\headheight}{14.49999pt}
}

\pagestyle{headings}
\setstretch{1.5}
}

% Reference commands
\newcommand{\Capitulo}[1] {\Cref{#1}}
\newcommand{\capitulos}[1]{\cref{#1}}
\newcommand{\Secao}[1]    {\Cref{#1}}
\newcommand{\secoes}[1]   {\cref{#1}}
\newcommand{\Figura}[1]   {\Cref{#1}}
\newcommand{\figuras}[1]  {\cref{#1}}
\newcommand{\Tabela}[1]   {\Cref{#1}}
\newcommand{\tabelas}[1]  {\cref{#1}}
\newcommand{\Equacao}[1]  {\Cref{#1}}
\newcommand{\equacoes}[1] {\cref{#1}}

% Common abbreviations
\newcommand{\ie}{i.\,e.\xspace}
\newcommand{\eg}{e.\,g.\xspace}

% Acronym commands
\newcommand{\Sigla}[2]{#1 (#2)\nomenclature{#2}{#1}}
\newcommand{\SiglaHifen}[2]{#1 -- #2\nomenclature{#2}{#1}}