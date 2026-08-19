"""Curated multi-disciplinary academic benchmark fixtures and mock faculty dataset."""

from typing import List, Dict, Any
from scholarmatch.models.schemas import FacultyProfile, ActiveGrant, Publication, CandidateProfile

BENCHMARK_FACULTY: List[FacultyProfile] = [
    FacultyProfile(
        id="fac-mit-csail-01",
        name="Prof. Regina Barzilay",
        institution="Massachusetts Institute of Technology (MIT)",
        department="EECS / CSAIL",
        lab_name="Barzilay Lab for Machine Learning in Health",
        lab_website="https://people.csail.mit.edu/regina/",
        research_summary=(
            "Developing deep learning models for drug discovery, molecular property prediction, and clinical diagnostics. "
            "Our research pioneers graph neural networks (GNNs) and generative geometric deep learning for de novo antibiotic design, "
            "cancer risk assessment, and molecular optimization."
        ),
        specialties=["Graph Neural Networks", "Drug Discovery", "Geometric Deep Learning", "Clinical Oncology AI", "Antibiotic Design"],
        h_index=84,
        accepting_students=True,
        active_grants=[
            ActiveGrant(
                grant_id="NSF-IIS-2309101",
                title="Geometric Graph Representation Learning for De Novo Molecular Design",
                agency="National Science Foundation (NSF)",
                amount_usd=1200000.0,
                start_year=2024,
                end_year=2027,
                abstract_or_summary="Investigating equivariance in 3D molecular graph convolutions for antibiotic synthesis and binding affinity optimization.",
                keywords=["graph neural networks", "molecular representation", "antibiotic design", "equivariance"]
            ),
            ActiveGrant(
                grant_id="NIH-R01-CA240112",
                title="Deep Geometric Generative Architectures for Cancer Risk Stratification",
                agency="National Institutes of Health (NIH)",
                amount_usd=2500000.0,
                start_year=2023,
                end_year=2028,
                abstract_or_summary="Machine learning models utilizing longitudinal mammography and molecular biomarkers for early tumor prediction.",
                keywords=["clinical AI", "biomarkers", "cancer prediction", "generative models"]
            )
        ],
        recent_publications=[
            Publication(
                title="A Deep Learning Approach to Antibiotic Discovery",
                abstract="Deep learning models can discover novel antibacterial molecules from massive chemical spaces without pre-engineered molecular fingerprints.",
                year=2020,
                venue="Cell",
                doi="10.1016/j.cell.2020.01.021",
                citation_count=2300,
                keywords=["antibiotic discovery", "deep learning", "graph neural networks", "molecular screening"],
                references=["10.1038/nature12345", "10.1126/science.1197258"]
            ),
            Publication(
                title="Equivariant Graph Neural Networks for 3D Macromolecular Complexes",
                abstract="Equivariant neural message passing preserves roto-translational symmetries when predicting protein-ligand binding poses and free energy affinities.",
                year=2023,
                venue="ICML",
                doi="10.48550/arXiv.2302.04321",
                citation_count=180,
                keywords=["equivariant networks", "molecular docking", "rotational symmetry", "protein-ligand"],
                references=["10.1016/j.cell.2020.01.021", "10.1038/s41586-021-03819-2"]
            )
        ]
    ),
    FacultyProfile(
        id="fac-stanford-ai-02",
        name="Prof. Percy Liang",
        institution="Stanford University",
        department="Computer Science",
        lab_name="Center for Research on Foundation Models (CRFM)",
        lab_website="https://crfm.stanford.edu/",
        research_summary=(
            "Holistic evaluation and reliable alignment of large language models (LLMs). "
            "We focus on benchmark robustness, uncertainty quantification, factual consistency, verifiable attribution, "
            "and computational efficiency in multi-step reasoning systems."
        ),
        specialties=["Large Language Models", "Holistic Evaluation", "Factual Consistency", "Prompt Engineering", "Uncertainty Estimation"],
        h_index=72,
        accepting_students=True,
        active_grants=[
            ActiveGrant(
                grant_id="DARPA-ANSR-2023",
                title="Verifiable Attribution and Formal Soundness in Neuro-Symbolic Foundation Models",
                agency="DARPA",
                amount_usd=3100000.0,
                start_year=2023,
                end_year=2026,
                abstract_or_summary="Developing mathematical guarantees and factual grounding algorithms for autonomous multi-step reasoning agents.",
                keywords=["foundation models", "verifiable attribution", "neuro-symbolic", "soundness"]
            )
        ],
        recent_publications=[
            Publication(
                title="Holistic Evaluation of Language Models (HELM)",
                abstract="A standardized multidimensional benchmarking framework evaluating foundation models across accuracy, robustness, fairness, and toxicity.",
                year=2023,
                venue="Transactions on Machine Learning Research",
                doi="10.48550/arXiv.2211.09110",
                citation_count=1200,
                keywords=["benchmarking", "foundation models", "robustness", "evaluation metrics"],
                references=["10.1145/3442188.3445922", "10.18653/v1/2020.emnlp-main.123"]
            ),
            Publication(
                title="Measuring and Improving Verifiable Citation in LLM Generation",
                abstract="Evaluating the exact factual grounding and citation precision of generative models against reference corpora to prevent hallucinations.",
                year=2024,
                venue="ACL",
                doi="10.18653/v1/2024.acl-long.456",
                citation_count=95,
                keywords=["factual grounding", "citation accuracy", "hallucination reduction", "verbatim evidence"],
                references=["10.48550/arXiv.2211.09110", "10.1145/3442188.3445922"]
            )
        ]
    ),
    FacultyProfile(
        id="fac-berkeley-robotics-03",
        name="Prof. Sergey Levine",
        institution="UC Berkeley",
        department="EECS",
        lab_name="Robotic AI & Learning (RAIL) Lab",
        lab_website="https://rail.eecs.berkeley.edu/",
        research_summary=(
            "Reinforcement learning and decision making for autonomous robotic systems. "
            "Our group investigates offline reinforcement learning, vision-language-action (VLA) foundation models, "
            "sim-to-real transfer, and whole-body dexterous manipulation in unstructured physical environments."
        ),
        specialties=["Offline Reinforcement Learning", "Robot Learning", "Vision-Language-Action Models", "Dexterous Manipulation", "Sim-to-Real"],
        h_index=115,
        accepting_students=True,
        active_grants=[
            ActiveGrant(
                grant_id="NSF-FRR-2208940",
                title="Scalable Cross-Embodiment Robot Learning from Massive Heterogeneous Datasets",
                agency="National Science Foundation (NSF)",
                amount_usd=1500000.0,
                start_year=2023,
                end_year=2026,
                abstract_or_summary="Synthesizing generalist robotic policies across varied arm and mobile robot morphologies with transformer architectures.",
                keywords=["robot learning", "cross-embodiment", "offline RL", "imitation learning"]
            )
        ],
        recent_publications=[
            Publication(
                title="Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems",
                abstract="Comprehensive mathematical framework for training policies from static logged data without ongoing environmental interaction.",
                year=2020,
                venue="arXiv preprint",
                doi="10.48550/arXiv.2005.01643",
                citation_count=3100,
                keywords=["offline RL", "reinforcement learning", "batch RL", "policy optimization"],
                references=["10.1038/nature14236", "10.1126/scirobotics.aau5872"]
            ),
            Publication(
                title="RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control",
                abstract="Co-fine-tuning large vision-language models on web-scale internet data and robotic trajectories enables emergent semantic reasoning in physical manipulation.",
                year=2023,
                venue="CoRL",
                doi="10.48550/arXiv.2307.15818",
                citation_count=650,
                keywords=["vision-language-action", "robotics", "foundation models", "manipulation"],
                references=["10.48550/arXiv.2005.01643", "10.48550/arXiv.2211.09110"]
            )
        ]
    ),
    FacultyProfile(
        id="fac-oxford-quantum-04",
        name="Prof. Vlatko Vedral",
        institution="University of Oxford",
        department="Clarendon Laboratory / Physics",
        lab_name="Oxford Quantum Information Theory Group",
        lab_website="https://www.physics.ox.ac.uk/research/group/quantum-information-theory",
        research_summary=(
            "Quantum thermodynamics, quantum entanglement measures, and quantum computing error mitigation. "
            "We explore the boundaries of macroscopic quantum coherence, quantum discord, tensor networks, and quantum machine learning algorithms."
        ),
        specialties=["Quantum Information", "Quantum Entanglement", "Quantum Thermodynamics", "Tensor Networks", "Quantum Error Mitigation"],
        h_index=79,
        accepting_students=True,
        active_grants=[
            ActiveGrant(
                grant_id="ERC-ADV-883902",
                title="Quantum Thermodynamic Machines and Non-Equilibrium Coherence Limits",
                agency="European Research Council (ERC)",
                amount_usd=2800000.0,
                start_year=2022,
                end_year=2027,
                abstract_or_summary="Investigating work extraction and entropy production in nanoscale quantum open systems.",
                keywords=["quantum thermodynamics", "open systems", "coherence", "entanglement"]
            )
        ],
        recent_publications=[
            Publication(
                title="Introduction to Quantum Information Science",
                abstract="Mathematical foundations of quantum bits, Bell inequalities, teleportation, and quantum computational complexity.",
                year=2021,
                venue="Oxford University Press",
                doi="10.1093/oso/9780199215706.001.0001",
                citation_count=1800,
                keywords=["quantum information", "entanglement", "quantum computing", "density matrices"],
                references=["10.1103/PhysRevLett.70.1895", "10.1103/RevModPhys.81.865"]
            ),
            Publication(
                title="Tensor Network Renormalization for Non-Equilibrium Quantum Phase Transitions",
                abstract="Matrix product states and tensor networks accurately simulate strongly correlated many-body systems under open dissipative dynamics.",
                year=2024,
                venue="Physical Review Letters",
                doi="10.1103/PhysRevLett.132.100401",
                citation_count=45,
                keywords=["tensor networks", "matrix product states", "dissipative systems", "phase transitions"],
                references=["10.1093/oso/9780199215706.001.0001", "10.1103/RevModPhys.81.865"]
            )
        ]
    ),
    FacultyProfile(
        id="fac-cmu-climate-05",
        name="Prof. Priya Donti",
        institution="Carnegie Mellon University (CMU)",
        department="Electrical and Computer Engineering & Public Policy",
        lab_name="Climate Change AI & Power System Optimization Lab",
        lab_website="https://www.priyadonti.com/",
        research_summary=(
            "Physics-informed machine learning and convex optimization for electric grid decarbonization and climate change mitigation. "
            "Our lab builds neural network architectures that strictly enforce physical laws, power flow constraints, and dynamical stability guarantees."
        ),
        specialties=["Physics-Informed ML", "Climate Change AI", "Power Grid Optimization", "Constrained Optimization", "Renewable Integration"],
        h_index=35,
        accepting_students=True,
        active_grants=[
            ActiveGrant(
                grant_id="DOE-EERE-390192",
                title="Physics-Constrained Deep Learning for Real-Time Distributed Renewable Grid Dispatch",
                agency="U.S. Department of Energy (DOE)",
                amount_usd=1950000.0,
                start_year=2024,
                end_year=2027,
                abstract_or_summary="Hard constraint satisfaction in deep neural networks managing volatility in wind and solar microgrid distribution.",
                keywords=["physics-informed ML", "power flow", "renewable energy", "constrained optimization"]
            )
        ],
        recent_publications=[
            Publication(
                title="Tackling Climate Change with Machine Learning",
                abstract="Comprehensive roadmap of high-leverage domains where machine learning can meaningfully reduce greenhouse gas emissions.",
                year=2021,
                venue="ACM Computing Surveys",
                doi="10.1145/3485128",
                citation_count=1600,
                keywords=["climate change AI", "smart grids", "carbon reduction", "machine learning"],
                references=["10.1038/s41558-019-0649-6", "10.1109/JPROC.2019.2941077"]
            ),
            Publication(
                title="DC3: Fast and Provably Feasible Constrained Optimization with Neural Networks",
                abstract="Implicit layer architectures that map infeasible predictions onto non-convex manifold boundaries to guarantee physical feasibility.",
                year=2023,
                venue="ICLR",
                doi="10.48550/arXiv.2104.12225",
                citation_count=130,
                keywords=["constrained optimization", "implicit layers", "physics guarantees", "power systems"],
                references=["10.1145/3485128", "10.1109/JPROC.2019.2941077"]
            )
        ]
    ),
    FacultyProfile(
        id="fac-eth-bio-06",
        name="Prof. Karsten Borgwardt",
        institution="ETH Zurich / Max Planck Institute",
        department="Biosystems Science and Engineering",
        lab_name="Machine Learning and Computational Biology Lab",
        lab_website="https://bsse.ethz.ch/mlcb",
        research_summary=(
            "Statistical machine learning and graph kernels for genome-wide association studies (GWAS) and biomarker discovery. "
            "We specialize in scalable hypothesis testing, combinatorial feature selection, and graph algorithms for biological networks."
        ),
        specialties=["Computational Biology", "Graph Kernels", "Biomarker Discovery", "GWAS", "Statistical Testing"],
        h_index=64,
        accepting_students=True,
        active_grants=[
            ActiveGrant(
                grant_id="SNF-197281",
                title="Combinatorial Graph Algorithms for Higher-Order Epistasis Detection in Genomic Cohorts",
                agency="Swiss National Science Foundation (SNSF)",
                amount_usd=1400000.0,
                start_year=2023,
                end_year=2026,
                abstract_or_summary="High-dimensional non-parametric statistical tests for gene-gene interactions in complex diseases.",
                keywords=["graph kernels", "genomics", "epistasis", "biomarkers"]
            )
        ],
        recent_publications=[
            Publication(
                title="Weisfeiler-Lehman Graph Kernels",
                abstract="Fast subtree kernels on graphs enabling high-accuracy classification and similarity calculation on chemical and biological graphs.",
                year=2011,
                venue="Journal of Machine Learning Research",
                doi="10.5555/1953048.2078187",
                citation_count=2900,
                keywords=["graph kernels", "weisfeiler-lehman", "graph similarity", "bioinformatics"],
                references=["10.1038/nature04567"]
            ),
            Publication(
                title="Graph Neural Networks in Computational Biology: A Rigorous Statistical Audit",
                abstract="Empirical and theoretical comparison between modern graph convolutional networks and classical Weisfeiler-Lehman graph kernels in small-sample bioinformatics benchmarks.",
                year=2024,
                venue="Bioinformatics",
                doi="10.1093/bioinformatics/btae120",
                citation_count=50,
                keywords=["graph neural networks", "statistical audit", "weisfeiler-lehman", "computational biology"],
                references=["10.5555/1953048.2078187", "10.1016/j.cell.2020.01.021"]
            )
        ]
    )
]

BENCHMARK_CANDIDATES: List[CandidateProfile] = [
    CandidateProfile(
        candidate_name="Alice Chen",
        thesis_title="Equivariant Graph Neural Networks for Targeted De Novo Antibiotic Discovery",
        statement_or_abstract=(
            "I am proposing a doctoral research program developing 3D equivariant geometric graph neural networks "
            "for molecular binding affinity prediction and automated de novo antibiotic design. My focus is on enforcing "
            "SE(3) rotational symmetries and physical constraints during molecular docking and generative screening against resistant bacterial pathogens."
        ),
        preferred_methods=["Graph Neural Networks", "Geometric Deep Learning", "Equivariance", "Generative Modeling"],
        target_domains=["Drug Discovery", "Antibiotic Resistance", "Molecular Chemistry"]
    ),
    CandidateProfile(
        candidate_name="Marcus Vance",
        thesis_title="Physics-Informed Deep Optimization for Renewable Power Grid Stability",
        statement_or_abstract=(
            "My research focuses on integrating physical power flow constraints into deep neural networks for real-time "
            "grid dispatch and renewable energy penetration. I design implicit layers with provable feasibility guarantees "
            "for non-convex optimal power flow problems under extreme climate uncertainty."
        ),
        preferred_methods=["Physics-Informed ML", "Constrained Optimization", "Implicit Layers", "Convex Relaxations"],
        target_domains=["Power Systems", "Climate Change AI", "Renewable Energy"]
    ),
    CandidateProfile(
        candidate_name="Sophia Rodriguez",
        thesis_title="Verifiable Citation and Exact Factual Provenance in Large Reasoning Agents",
        statement_or_abstract=(
            "My thesis investigates deterministic factual grounding and span-level citation attribution for foundation models. "
            "I design neuro-symbolic audit pipelines to compute exact string containment, longest common subsequence alignment, "
            "and bibliographic coupling networks to eliminate AI hallucinations in automated scientific review."
        ),
        preferred_methods=["Verifiable Attribution", "Neuro-Symbolic AI", "Bibliometric Graphs", "Exact String Alignment"],
        target_domains=["Scientific NLP", "Fact Verification", "Foundation Models"]
    )
]
