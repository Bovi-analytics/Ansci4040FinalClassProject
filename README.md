# ansci_4040_final_class_project
Repo for the final class project for ANSCI 4040.

## Project Goal:
The goal is to create a model that uses video data and computer vision alone to recognize at least one cow from Pen 51 entering the milking parlor from a group of cows (around 10) that calved during one day of the previous week. 

## Development Environment
This project will use Google Colab as a cloud-based development environment for data processing and model development. The Github repository will be the main source of coding collaboration between classmates.

## Strategy:
We split into 3 groups to divide the tasks:

Team Sloth: This group created a pipeline for the calving pen videos. This included watching raw videos and processing them (cutting when cows aren't in the frame, cutting hours of cows staying still, etc.) and creating a cluster based on the cow's different angles. They extracted features from each cow in the calving pens and created an embedding pool for each animal. 

Team Cow: This group was originally responsible for creating a plan for data organization, including a GitHub repository with a README file, data agreement, and a license. Then they built the pipeline for the milking parlor videos, sharing code from Team Sloth, chopping videos into smaller images, clustering them to create an embedding pool, and grouping similar images. They will also label selected images with the corresponding cow IDs and use the labeled images and embeddings to develop and test a model that identifies individual cows. 

Team Vacca: This group was tasked with collecting BoviSync reports to match cows between the milking parlor and calving area videos. Aside from collecting a universal order of the cows entering the milking parlor, this team was also responsible for determining necessary camera angle changes needed by both other groups and communicating decisions with the dairy manager. This group visited the farm as needed and assisted the other teams whenever possible.  

## Timeline

**Start date:** October 27

| Week | Task |
|---|---|
| Week 0 | Develop the project plan. |
| Week 1 | Finalize the camera setup. |
| Week 2 | Finish the labels. |
| Week 3 | Connect the calving area labels with the milking parlor labels. |
| Week 4 | Finalize the project. |

**Tentative Project** Due Date: November 24th @ 4:30 PM
**Project Presentation**: December 2nd @ 10:10 AM

## Final deliverable and assessment
The success of this project should be judged based on completion of the project goal and the quality of final presentation delivery

**Project Completion**: This project will be considered complete when the goal has been achieved. That is, a model is built and trained to identify cows that calved during a one-day period from three calving pens and can recognize at least one of these cows entering the milking parlor from Pen 51, using computer vision alone. 

**Presentation**: 
In the last week of class, we will give a 20-minute presentation to the course instructor, lab, and invited ARD members, followed by a Q&A. 
This presentation will be assessed by the following criteria:
- Statement of the project background, significance, and goal
- An outline of group division and strategy
- A brief overview from each group of procedural steps taken to achieve their team goal
- Discussion of model performance and limitations
- Future applications
IMPORTANTLY: each project member should have sufficient theoretical understanding of the project, including portions their team was not assigned to. Every member should participate (speak) during the presentation and question answering.
- A peer assessment may also be included in the final grade.  
