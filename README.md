<!--
MIT License

Copyright (c) 2018 Othneil Drew

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** marakeby, pnet_prostate_paper, twitter_handle, email, P-NET, project_description
-->





<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/Paureel/get_me_a_nobel_prize_streamlit">
    <img src="logo.png" alt="Logo" width="80" height="80">
  </a>
  
# Get Me A Nobel Prize - powered by LLMs
 A streamlined interface built on Streamlit that interacts with the GPT-4 or GPT-3.5 models (custom models coming soon!) from OpenAI to generate scientific hypotheses. Users can input their scientific problems or the output of their machine learning algorithm of choice to generate potential solutions and visualizations. The visualization is powered by UMAP, and is used to visualize the one-sentence embeddings of the generated hypotheses, where semanticaly similar hypotheses are going to be closer together.
 ![App Screenshot](screenshot.png)
 The app can be accessed here:
 
 ## 🌟 Features

- **Interactive UI**: Built with Streamlit.
- **OpenAI Integration**: Uses GPT-3.5-turbo and GPT-4 models.
- **Visualization**: UMAP-based scatter plot of one-sentence embeddings of the generated hypotheses.
- **AI Chat**: Chat with the model, view history, and download the chat history.

## 🔧 Prerequisites

- Python (>= 3.7)
- Streamlit
- OpenAI Python client
- UMAP
- Plotly

## 🚀 Installation & Usage

1. **Clone**:
   ```sh
   git clone https://github.com/Paureel/get_me_a_nobel_prize_streamlit
   ```
2. **Navigate**:
	```sh
	cd get_me_a_nobel_prize_streamlit
	```
3. **Install Dependencies**:
	```sh
	conda create --name envname --file requirements.txt
	```
4. **Run**:
	```sh
	streamlit run app.py
	```
	
## Todos

- [] Add custom prompts (the one I'm using is hardcoded)
- [] Ability to download the embedding vectors
- [] Add custom models instead of OpenAI models
- [] More examples
## ✨ Contributing

1. Fork the project.
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add some AmazingFeature'`
4. Push: `git push origin feature/AmazingFeature`
5. Open a pull request.

## 📄 License

MIT License. See `LICENSE` for details.

## Support

If you liked this project, you can support me by buying me a coffee here :) : buymeacoffee.com/aurelproszw

## Acknowledgements
Thank you for Astropomeai for the initial Tree of Thoughts Langchain implementation: https://medium.com/@astropomeai/implementing-the-tree-of-thoughts-in-langchains-chain-f2ebc5864fac