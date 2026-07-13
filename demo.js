const examples = [
  {
    question: "支持向量机如何找到最优分类边界？",
    route: "detail · Classification",
    answer: "支持向量机通过最大化两类样本与决策超平面之间的间隔来寻找分类边界。距离边界最近的样本称为支持向量，它们决定了超平面的位置。[S1] 对于不可线性分离的数据，可以引入软间隔与核函数。[S2]",
    source: "02-Classification/03-SupportVectorMachines.md",
  },
  {
    question: "K-means 聚类的步骤是什么？",
    route: "detail · Clustering",
    answer: "K-means 先选择 K 个初始质心，再把每个样本分配给最近的质心，并用簇内样本均值更新质心。分配与更新交替进行，直到质心或目标函数收敛。[S1]",
    source: "03-Clustering/01-K-meansClustering.md",
  },
  {
    question: "线性回归有哪些假设？",
    route: "detail · Regression",
    answer: "线性回归通常假设线性关系、误差独立、同方差、残差近似正态，并避免严重多重共线性。[S1] 这些假设决定了系数估计与统计推断是否可靠。",
    source: "01-Regression/01-LinearRegression.md",
  },
];

const buttons = [...document.querySelectorAll(".exampleTabs button")];
const question = document.querySelector(".queryBlock h2");
const route = document.querySelector(".trace:first-child p");
const answer = document.querySelector(".answerBlock > p");
const source = document.querySelector(".source");

buttons.forEach((button, index) => {
  button.addEventListener("click", () => {
    const item = examples[index];
    buttons.forEach((candidate, candidateIndex) => {
      candidate.setAttribute("aria-pressed", String(candidateIndex === index));
    });
    question.firstChild.textContent = item.question;
    route.textContent = item.route;
    answer.textContent = item.answer;
    source.replaceChildren(Object.assign(document.createElement("span"), { textContent: "[S1]" }), ` ${item.source}`);
  });
});
