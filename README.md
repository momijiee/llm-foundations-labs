# LLM Foundations Labs

An evolving collection of reproducible experiments and application studies for **Exploration of Large Model Fundamentals and Applications**.

This repository is designed as a course-long technical portfolio. Each lab records the question being investigated, the implementation, the experiment setup, the evidence produced, and the limitations observed. The emphasis is on understanding model behavior and making results reproducible—not merely calling an API.

## Course map

The course progresses from neural-network foundations to language models, multimodal systems, generative models, application development, and responsible AI. This repository will grow along the same path.

| Area | Planned evidence in this repository |
| --- | --- |
| Neural-network foundations | Handwritten PyTorch training loops and controlled optimization studies |
| Language models and text applications | Prompting, evaluation, and task-oriented prototypes |
| Multimodal and visual generation | Small, documented image or multimodal experiments where applicable |
| LLM applications | Reproducible implementations, comparison criteria, and failure analysis |
| Responsible AI | Limitations, safety, and ethical considerations recorded with each relevant project |

Only completed work is listed below. Future directories will be added together with the corresponding assignment, rather than as empty placeholders.

## Completed labs

| Lab | Focus | Key result |
| --- | --- | --- |
| [01 — MLP digits experiment](labs/01-mlp-digits/) | Build and train an MLP in PyTorch from first principles | ReLU + Adam + BatchNorm achieved 95.83% test accuracy in three verification runs |
| [02 — ConvLSTM bouncing balls](labs/02-convlstm-bouncing-balls/) | Controlled spatiotemporal next-frame prediction | A combined ConvLSTM configuration reached 0.003967 mean test MSE across three runs |

## Repository layout

```text
.
├── labs/
│   ├── 01-mlp-digits/       # A self-contained experiment: code, notebook, results, and notes
│   └── 02-convlstm-bouncing-balls/
├── README.md                 # Course-level overview and learning trajectory
└── LICENSE
```

## Reproducibility principles

- Keep each lab self-contained with its own setup notes and dependencies.
- Fix random seeds and record the evaluation protocol for comparative experiments.
- Publish code and curated results, but keep submitted reports, course handouts, and personal information out of the repository.
- Document negative results and limitations when they improve the explanation of a method.

## About

Maintained by [momijiee](https://github.com/momijiee), an undergraduate computer science student at Harbin Institute of Technology, Shenzhen.

## License

Released under the [MIT License](LICENSE).
