import model_utils.solver as sovler

class FCN_sovler(sovler.solver):
    def __init__(self, models, model_name, save_path="checkpoints"):
        super(FCN_sovler,self).__init__(models,model_name,save_path)
        self.fcn=self.models[0]

    def train_one_batch(self,input_dict):
        optimizer=self.optimizers[0]
        x=input_dict["x"]
        y=input_dict["y"]
        out=self.fcn(x)
        loss=torch.abs(x-out).mean()
        loss.backward()
        optimizer.step()
        self.zero_grad_for_all()
        total_loss={}
        total_loss["reconst loss"]=loss.detach().cpu().item()
        return total_loss

    def test_one_batch(self,input_dict):
        x=input_dict["x"]
        out=self.fcn(x)
        out[out<0]=0.0
        out[out>1]=1.0
        return out


    def train_loop(self,param_dict,epoch=10000):
        iteration_count=0
        dataloader=param_dict["loader"]
        for i in range(0,epochs):
            for step,(x,y) in enumerate(dataloader):
                input_dict={}
                input_dict["x"]=x
                input_dict["y"]=y
                loss=self.train_one_batch(input_dict)
                iteration_count+=1
                if(iteration_count%1==0):
                    self.write_log(loss,iteration_count)
                    self.output_loss(loss,i,iteration_count)
                if(iteration_count%100==0):
                    out_images=self.test_one_batch(input_dict)
                    images={}
                    images["image"]=out_images
                    self.write_log_image(images,int(iteration_count/100))
            if(i%50==0):
                self.save_models(epoch=i)
