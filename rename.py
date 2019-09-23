import tensorflow as tf

def rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
        print(var_name)
    print("!!!numbers of variables", len(tf.contrib.framework.list_variables(checkpoint_dir)))
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            # Set the new name
            new_name = var_name
            new_name = new_name.replace(replace_from, replace_to)
            var_gen = tf.Variable(var, name=var_name)   
            if replace_from in var_name:
                print('Renaming %s to %s.' % (var_name, new_name))
                # Rename the variable
                var_dis = tf.Variable(var, name=new_name)
                #var = tf.Variable(var, name=var_name)

        # Save the variables
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, checkpoint.model_checkpoint_path)
        print("rename variable name, and saving in", checkpoint.model_checkpoint_path)

if __name__ == "__main__":
   import os
   os.environ["CUDA_VISIBLE_DEVICES"] = "6"
   checkpoint_dir = "/home/work/xiepan/xp_dial/gan_nmt/transformer_rl/test/train_small"
   rename(checkpoint_dir, "Transforme", "Discriminator", None , False)
   for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
       print(var_name)
