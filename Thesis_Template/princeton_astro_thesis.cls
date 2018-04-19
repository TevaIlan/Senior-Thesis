% Class file for Princeton University senior theses and junior papers
% Written for the Astrophysics Department

% Setup
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{princeton_astro_thesis}[2013/11/10 Princeton Thesis]

% Option for toggling JP version
\DeclareOption{jp}{\def\@juniorpaper{}}

% All other options
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}

% Process options
\ProcessOptions\relax

% Load report class
\LoadClass{report}

% Load packages
\RequirePackage[T1]{fontenc}
\RequirePackage[margin=1in,inner=1.5in]{geometry}
\RequirePackage{setspace}

% Provide commands for user input
\long\def\title#1{\def\@title{#1}}
\long\def\author#1{\def\@author{#1}}
\long\def\adviser#1{\def\@adviser{#1}}
\long\def\date#1{\def\@date{#1}}
\long\def\abstract#1{\def\@abstract{#1}}
\long\def\acknowledgments#1{\def\@acknowledgments{#1}}
\long\def\dedication#1{\def\@dedication{#1}}

% Default empty inputs where field should be displayed anyway
\def\@abstract{}
\def\@title{}
\def\@author{}
\def\@adviser{}
\def\@date{}

% Default fields that need not be changed
\def\@thesistype{ %
  \ifx\@juniorpaper\undefined %
    Senior Thesis %
  \else %
    Junior Paper %
  \fi
}
\def\@academicdegree{Bachelor of the Arts}
\def\@department{Astrophysical Sciences}
\def\@university{Princeton University}

% Define submission and acceptance text
\long\def\@submissiontext{ %
  A \@thesistype \\ %
  presented to the faculty \\ %
  of \@university \\ %
  in candidacy for the degree \\ %
  of \@academicdegree
}
\long\def\@acceptancetext{ %
  Recommended for acceptance \\ %
  by the Department of \\ %
  \@department
}

% Define titlepage
\renewcommand{\maketitle}{{ %
  \thispagestyle{empty} %
  \large\scshape\centering %
  {\Large \@title} \\[\stretch{1}] %
  \@author \\[\stretch{1}] %
  \@submissiontext \\[\stretch{1}] %
  \@acceptancetext \\[\stretch{1}] %
  Adviser: \@adviser \\[\stretch{1}] %
  \@date
}}

% Define authorization page
\newcommand{\makeauthorization}{ %
  \vspace*{\stretch{1}} %
  \noindent I hereby declare that I am the sole author of this thesis. \\[0.25in] %
  \noindent I authorize Princeton University to lend this thesis to other institutions or individuals for the purpose of scholarly research. \\[0.5in] %
  \null\hfill \makebox[2in][l]{\hrulefill} \\ %
  \null\hfill \makebox[2in][l]{\@author} \\[0.5in] %
  \noindent I further authorize Princeton University to reproduce this thesis by photocopying or by other means, in total or in part, at the request of other institutions or individuals for the purpose of scholarly research. \\[0.5in] %
  \null\hfill \makebox[2in][l]{\hrulefill} \\ %
  \null\hfill \makebox[2in][l]{\@author}
  \vspace*{\stretch{4}} %
}

% Define abstract
\newcommand{\makeabstract}{ %
  \addcontentsline{toc}{section}{Abstract} %
  \section*{Abstract} %
  \ifx\@abstract\undefined %
  \else %
    \@abstract %
  \fi
}

% Define acknowledgments
\newcommand{\makeacknowledgments}{ %
  \ifx\@acknowledgments\undefined %
  \else %
    \addcontentsline{toc}{section}{Acknowledgments} %
    \section*{Acknowledgments} %
    \@acknowledgments %
  \fi
}

% Define dedication
\newcommand{\makededication}{ %
  \ifx\@dedication\undefined %
  \else %
    \vspace*{\stretch{1}} %
    \begin{flushright} %
      \@dedication %
    \end{flushright} %
    \vspace*{\stretch{3}} %
  \fi
}

% Set up document
\AtBeginDocument{ %
  \pagenumbering{roman} %
  \maketitle \clearpage %
  \makeauthorization \clearpage %
  \makeabstract \clearpage %
  \makeacknowledgments \clearpage %
  \makededication \clearpage %
  \tableofcontents \clearpage %
  \pagenumbering{arabic}
}

\endinput
